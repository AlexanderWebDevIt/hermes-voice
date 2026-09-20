/**
 * Воспроизведение ответа.
 *
 * Сервер режет ответ на предложения и присылает каждое отдельным потоком аудио,
 * помечая начало событием `speech_chunk`. Здесь эти потоки собираются в файлы
 * и играются по очереди — так первый звук раздаётся, пока модель ещё печатает
 * остаток ответа, а не после того, как всё готово.
 *
 * Формат приходит от сервера в `hello` и зависит от движка синтеза:
 * локальный piper отдаёт wav, облачный edge — mp3. Расширение файла важно:
 * expo-audio определяет кодек в том числе по нему.
 */

import * as FileSystem from 'expo-file-system/legacy';
import { createAudioPlayer, setAudioModeAsync, type AudioPlayer } from 'expo-audio';

/** Собрать base64 из байтов (порциями, чтобы не переполнить стек на больших кусках). */
function bytesToBase64(bytes: Uint8Array): string {
  const CHUNK = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return globalThis.btoa ? globalThis.btoa(binary) : '';
}

/** Соединить несколько кусков в один массив. */
function concat(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

export interface PlayerCallbacks {
  /** Началось воспроизведение очередного куска. */
  onChunkStart?: (text: string) => void;
  /** Всё, что было в очереди, доиграно. */
  onIdle?: () => void;
  onError?: (message: string) => void;
}

export class ReplyPlayer {
  private player: AudioPlayer | null = null;
  private queue: Array<{ uri: string; text: string }> = [];
  private pending: Uint8Array[] = [];
  private pendingText = '';
  private seq = 0;
  private playing = false;
  private generation = 0;
  private disposed = false;
  /** Расширение файлов: приходит в hello от сервера. */
  private extension = 'wav';

  constructor(private callbacks: PlayerCallbacks = {}) {}

  /** Сервер сказал, в каком формате будет аудио (wav у piper, mp3 у edge). */
  setFormat(format: string): void {
    this.extension = format === 'mp3' ? 'mp3' : 'wav';
  }

  /** Подготовить аудиорежим: звук должен играть и в беззвучном режиме телефона. */
  async init(): Promise<void> {
    try {
      await setAudioModeAsync({
        playsInSilentMode: true,
        shouldPlayInBackground: false,
        allowsRecording: true,
      });
    } catch (error) {
      this.callbacks.onError?.(`Аудиорежим: ${String(error)}`);
    }
  }

  /** Сервер сообщил о начале нового куска — закрываем предыдущий. */
  async beginChunk(text: string): Promise<void> {
    await this.flush();
    this.pendingText = text;
  }

  /** Очередной бинарный фрейм текущего куска. */
  pushAudio(bytes: Uint8Array): void {
    this.pending.push(bytes);
  }

  /** Закрыть текущий кусок и поставить его в очередь на воспроизведение. */
  async flush(): Promise<void> {
    if (!this.pending.length) return;

    const data = concat(this.pending);
    const text = this.pendingText;
    this.pending = [];
    this.pendingText = '';

    const uri = `${FileSystem.cacheDirectory}hv_reply_${this.seq++}.${this.extension}`;
    try {
      await FileSystem.writeAsStringAsync(uri, bytesToBase64(data), {
        encoding: FileSystem.EncodingType.Base64,
      });
    } catch (error) {
      this.callbacks.onError?.(`Не удалось сохранить звук: ${String(error)}`);
      return;
    }

    this.queue.push({ uri, text });
    void this.playNext();
  }

  /** Проиграть следующий кусок, если сейчас ничего не играет. */
  private async playNext(): Promise<void> {
    if (this.playing || this.disposed) return;
    const item = this.queue.shift();
    if (!item) {
      this.callbacks.onIdle?.();
      return;
    }

    this.playing = true;
    const generation = this.generation;
    this.callbacks.onChunkStart?.(item.text);

    try {
      this.player?.remove();
      const player = createAudioPlayer({ uri: item.uri });
      this.player = player;

      player.addListener('playbackStatusUpdate', (status) => {
        if (generation !== this.generation) return; // кусок уже отменён
        if (status.didJustFinish) {
          void this.cleanup(item.uri);
          this.playing = false;
          void this.playNext();
        }
      });

      player.play();
    } catch (error) {
      this.callbacks.onError?.(`Не удалось воспроизвести: ${String(error)}`);
      this.playing = false;
      void this.cleanup(item.uri);
      void this.playNext();
    }
  }

  private async cleanup(uri: string): Promise<void> {
    try {
      await FileSystem.deleteAsync(uri, { idempotent: true });
    } catch {
      // временный файл удалится системой — это не ошибка сценария
    }
  }

  /** Прервать всё: barge-in или остановка сессии. */
  async stop(): Promise<void> {
    this.generation += 1;   // отсекаем колбэки от старых кусков
    this.playing = false;
    this.pending = [];
    this.pendingText = '';

    try {
      this.player?.pause();
      this.player?.remove();
    } catch {
      // плеер мог быть уже освобождён
    }
    this.player = null;

    const stale = this.queue;
    this.queue = [];
    await Promise.all(stale.map((item) => this.cleanup(item.uri)));
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    await this.stop();
  }
}
