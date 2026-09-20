/**
 * Протокол обмена с hermes-voice.
 * Описание целиком — в README сервиса.
 */

/** Состояние диалога, которое присылает сервер. */
export type VoiceState = 'idle' | 'listening' | 'thinking' | 'speaking';

/**
 * Бэкенд ответов.
 *  - `direct` — прямой вызов модели: быстро (первый звук ~3-5 с), но без инструментов.
 *  - `agent`  — полный Hermes Agent: умеет искать в вебе и читать файлы, но отвечает
 *               в разы дольше (первый звук 10-20 с).
 */
export type Backend = 'direct' | 'agent';

/** Формат аудио, которое отдаёт сервер: wav от локального piper, mp3 от облачного edge. */
export interface OutputFormat {
  format: 'wav' | 'mp3';
  sample_rate: number;
  channels: number;
  engine: string;
  voices: string[];
}

export interface HelloEvent {
  type: 'hello';
  protocol: number;
  input: { format: string; sample_rate: number; channels: number };
  output: OutputFormat;
  voice: string;
  backend: Backend;
  backends: Backend[];
  stt_model: string;
}

export interface StateEvent {
  type: 'state';
  value: VoiceState;
}

export interface LevelEvent {
  type: 'level';
  value: number;
}

export interface TranscriptEvent {
  type: 'transcript';
  text: string;
  final: boolean;
}

export interface AssistantDeltaEvent {
  type: 'assistant_delta';
  text: string;
}

export interface AssistantDoneEvent {
  type: 'assistant_done';
  text: string;
}

/**
 * Начало очередного куска озвучки: все бинарные фреймы до следующего маркера — его звук.
 *
 * `filler: true` помечает отбивку — короткое «так…», которое сервер вставляет,
 * если модель думает дольше порога. Это не часть ответа, поэтому `text` пуст
 * и `seq` равен -1.
 */
export interface SpeechChunkEvent {
  type: 'speech_chunk';
  seq: number;
  text: string;
  filler?: boolean;
}

export interface ErrorEvent {
  type: 'error';
  message: string;
}

export interface PongEvent {
  type: 'pong';
}

export interface ConfigEvent {
  type: 'config';
  voice?: string;
  backend?: Backend;
}

export type ServerEvent =
  | HelloEvent
  | StateEvent
  | LevelEvent
  | TranscriptEvent
  | AssistantDeltaEvent
  | AssistantDoneEvent
  | SpeechChunkEvent
  | ErrorEvent
  | PongEvent
  | ConfigEvent;

/** Команды, которые шлёт приложение. */
export type ClientCommand =
  | { type: 'start'; voice?: string; sample_rate?: number }
  | { type: 'stop' }
  | { type: 'interrupt' }
  | { type: 'text'; content: string }
  | { type: 'config'; voice?: string; backend?: Backend }
  | { type: 'ping' };
