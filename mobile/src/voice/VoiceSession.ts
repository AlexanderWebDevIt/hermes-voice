/**
 * Голосовая сессия: микрофон → сервер → динамик.
 *
 * Класс держит WebSocket, гоняет аудио в обе стороны и отдаёт наружу события
 * для интерфейса. Всё состояние диалога (кто когда говорит) определяет сервер —
 * клиент только передаёт звук и рисует то, что пришло.
 */

import { ReplyPlayer } from '../audio/player';
import type { Backend, ClientCommand, ServerEvent, VoiceState } from './protocol';

export interface SessionSnapshot {
  connected: boolean;
  state: VoiceState;
  /** Распознанная речь пользователя. */
  transcript: string;
  /** Ответ, накапливается по мере стриминга. */
  answer: string;
  /** Текущий уровень микрофона, 0..1. */
  level: number;
  error: string | null;
  /** Сервер прислал hello — соединение готово к работе. */
  ready: boolean;
  /** Активный бэкенд ответов. */
  backend: Backend;
  /** Какие бэкенды доступны на сервере. */
  backends: Backend[];
  /** Голоса, доступные на сервере. */
  voices: string[];
  /** Активный голос. */
  voice: string;
  /** Микрофон должен быть включён — на это реагирует экран, поднимая поток. */
  micWanted: boolean;
  /** Фактическая частота микрофона, сообщённая телефоном. */
  micRate: number | null;
}

type Listener = (snapshot: SessionSnapshot) => void;

const INITIAL: SessionSnapshot = {
  connected: false,
  state: 'idle',
  transcript: '',
  answer: '',
  level: 0,
  error: null,
  ready: false,
  backend: 'direct',
  backends: ['direct'],
  voices: [],
  voice: '',
  micWanted: false,
  micRate: null,
};

/** Задержка перед переподключением, мс. */
const RECONNECT_DELAY = 2500;

export class VoiceSession {
  private ws: WebSocket | null = null;
  private snapshot: SessionSnapshot = { ...INITIAL };
  private listeners = new Set<Listener>();
  private player = new ReplyPlayer({
    onError: (message) => this.patch({ error: message }),
  });
  private url = '';
  private voice = '';
  private wantListening = false;
  /** Сервер уже получил команду start для текущего сеанса слушания. */
  private started = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private closedByUser = false;

  // ------------------------------------------------------------------ подписка

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.snapshot);
    return () => this.listeners.delete(listener);
  }

  private patch(partial: Partial<SessionSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...partial };
    for (const listener of this.listeners) listener(this.snapshot);
  }

  getSnapshot(): SessionSnapshot {
    return this.snapshot;
  }

  // ------------------------------------------------------------------ соединение

  async connect(url: string, voice: string): Promise<void> {
    this.url = url;
    this.voice = voice;
    this.closedByUser = false;
    await this.player.init();
    this.open();
  }

  private open(): void {
    this.clearReconnect();
    try {
      const ws = new WebSocket(this.url);
      ws.binaryType = 'arraybuffer';
      this.ws = ws;

      ws.onopen = () => {
        this.patch({ connected: true, error: null });
        // Настройки шлём до start: сервер должен знать бэкенд и голос до первой фразы.
        this.send({ type: 'config', voice: this.voice, backend: this.snapshot.backend });
        // Если слушание уже идёт, после переподключения ждём первый буфер —
        // он заново сообщит серверу частоту микрофона.
        this.started = false;
      };

      ws.onmessage = (event) => void this.onMessage(event);

      ws.onerror = () => {
        this.patch({ error: 'Нет связи с сервером', connected: false });
      };

      ws.onclose = () => {
        this.patch({ connected: false, ready: false, state: 'idle', micRate: null });
        this.started = false;
        if (!this.closedByUser) this.scheduleReconnect();
      };
    } catch (error) {
      this.patch({ error: `Не удалось подключиться: ${String(error)}` });
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer || this.closedByUser) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.open();
    }, RECONNECT_DELAY);
  }

  private clearReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  async disconnect(): Promise<void> {
    this.closedByUser = true;
    this.wantListening = false;
    this.started = false;
    this.clearReconnect();
    await this.player.dispose();
    try {
      this.ws?.close();
    } catch {
      // соединение могло уже закрыться
    }
    this.ws = null;
    this.patch({ ...INITIAL });
  }

  // ------------------------------------------------------------------ команды

  private send(command: ClientCommand): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(command));
    }
  }

  /** Начать слушать микрофон. */
  async startListening(): Promise<void> {
    this.wantListening = true;
    // `start` серверу пока не шлём: сначала нужно узнать фактическую частоту
    // микрофона, иначе он будет резать поток по неверным порогам. Отправит его
    // pushAudio() на первом же буфере.
    this.started = false;
    this.patch({ transcript: '', answer: '', error: null, micWanted: true });
    this.startMic();
  }

  /** Перестать слушать (сессия остаётся открытой). */
  async stopListening(): Promise<void> {
    this.wantListening = false;
    this.send({ type: 'stop' });
    this.stopMic();
    await this.player.stop();
    this.patch({ state: 'idle' });
  }

  /** Прервать ответ агента. */
  async interrupt(): Promise<void> {
    this.send({ type: 'interrupt' });
    await this.player.stop();
  }

  /** Сменить голос на лету. */
  setVoice(voice: string): void {
    this.voice = voice;
    this.patch({ voice });
    this.send({ type: 'config', voice });
  }

  /**
   * Переключить бэкенд ответов.
   * `direct` — быстро, без инструментов; `agent` — полный Hermes, но медленно.
   */
  setBackend(backend: Backend): void {
    this.patch({ backend });
    this.send({ type: 'config', backend });
  }

  /** Отправить текст без голоса. */
  sendText(content: string): void {
    this.patch({ answer: '', transcript: content });
    this.send({ type: 'text', content });
  }

  // ------------------------------------------------------------------ микрофон

  private startMic(): void {
    // Потоком управляет экран (хук useMicrophone) — здесь только флаг желания.
  }

  private stopMic(): void {
    this.patch({ level: 0, micWanted: false });
  }

  /**
   * Порция PCM от микрофона.
   *
   * Первый буфер заодно сообщает серверу фактическую частоту дискретизации —
   * до него слать `start` бессмысленно, сервер не знает, как резать поток.
   */
  pushAudio(pcm: ArrayBuffer, sampleRate: number): void {
    const ws = this.ws;
    if (ws?.readyState !== WebSocket.OPEN) return;

    if (!this.started) {
      this.started = true;
      this.patch({ micRate: sampleRate });
      this.send({ type: 'start', voice: this.voice, sample_rate: sampleRate });
    }
    ws.send(pcm);
  }

  // ------------------------------------------------------------------ приём

  private async onMessage(event: WebSocketMessageEvent): Promise<void> {
    if (typeof event.data !== 'string') {
      if (event.data instanceof ArrayBuffer) {
        this.player.pushAudio(new Uint8Array(event.data));
      }
      return;
    }

    let payload: ServerEvent;
    try {
      payload = JSON.parse(event.data) as ServerEvent;
    } catch {
      return;
    }

    switch (payload.type) {
      case 'hello': {
        // Формат аудио задаёт сервер: от него зависит расширение файла для плеера.
        this.player.setFormat(payload.output.format);

        // Выбор человека важнее серверного, но только если такой голос на
        // сервере действительно есть. Сохранённый голос может остаться от
        // другого движка (edge-tts ↔ Piper) — тогда его надо заменить, иначе
        // сервер ответит «Неизвестный голос» и озвучки не будет вовсе.
        const known = payload.output.voices;
        if (!this.voice || (known.length > 0 && !known.includes(this.voice))) {
          this.voice = payload.voice;
        }

        this.patch({
          ready: true,
          error: null,
          backend: payload.backend,
          backends: payload.backends,
          voices: payload.output.voices,
          voice: this.voice,
        });
        break;
      }

      case 'state':
        this.patch({ state: payload.value });
        break;

      case 'level':
        this.patch({ level: payload.value });
        break;

      case 'transcript':
        this.patch({ transcript: payload.text });
        break;

      case 'assistant_delta':
        this.patch({ answer: this.snapshot.answer + payload.text });
        break;

      case 'assistant_done':
        this.patch({ answer: payload.text });
        break;

      case 'speech_chunk':
        // Отбивка обрабатывается так же: её аудио просто встаёт в очередь перед
        // ответом. Текст ответа приходит отдельно, через assistant_delta, так что
        // пустой text у отбивки ничего не ломает.
        await this.player.beginChunk(payload.text);
        break;

      case 'error':
        this.patch({ error: payload.message });
        break;

      case 'config':
        this.patch({
          ...(payload.voice ? { voice: payload.voice } : {}),
          ...(payload.backend ? { backend: payload.backend } : {}),
        });
        break;

      default:
        break;
    }
  }
}
