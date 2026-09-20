/**
 * Захват микрофона: сырой PCM, чанками.
 *
 * Именно поток, а не запись в файл — иначе не получится живой диалог: сервер
 * должен слышать речь по мере поступления и сам понимать, когда фраза закончилась.
 *
 * Раньше здесь был `react-native-live-audio-stream`. Он даёт base64-строки,
 * не имеет типов и, главное, это bare-модуль: приложение с ним нельзя запустить
 * в Expo Go, только полной сборкой. `expo-audio` из SDK даёт то же самое
 * нативно — готовый `ArrayBuffer` с int16 и фактическую частоту дискретизации.
 */

import { useAudioStream } from 'expo-audio';
import { requestRecordingPermissionsAsync } from 'expo-audio';
import { useEffect, useRef, useState } from 'react';

/** Формат, который ждёт сервер. Менять только вместе с настройками hermes-voice. */
export const SAMPLE_RATE = 16000;
export const CHANNELS = 1;

/** Спросить разрешение на микрофон. */
export async function requestMicPermission(): Promise<boolean> {
  const response = await requestRecordingPermissionsAsync();
  return response.granted;
}

export interface MicrophoneOptions {
  /** Слушать или нет. Управляется состоянием диалога. */
  enabled: boolean;
  /** Очередная порция PCM s16le, моно, на частоте `sampleRate`. */
  onChunk: (pcm: ArrayBuffer, sampleRate: number) => void;
  onError?: (message: string) => void;
}

/**
 * Хук микрофона. Отдаёт PCM по мере захвата, пока `enabled` истинно.
 *
 * Частоту не фиксируем: телефон вправе отдать 48 кГц вместо запрошенных 16, и
 * `expo-audio` честно сообщает фактическую в каждом буфере. Приведение к 16 кГц
 * делает сервер — клиент остаётся тонким.
 */
export function useMicrophone({ enabled, onChunk, onError }: MicrophoneOptions) {
  const [actualRate, setActualRate] = useState<number | null>(null);

  // Колбэк держим в ref: хук подписывается на него один раз, а проп может меняться.
  const chunkRef = useRef(onChunk);
  chunkRef.current = onChunk;

  const { stream, isStreaming } = useAudioStream({
    sampleRate: SAMPLE_RATE,
    channels: CHANNELS,
    encoding: 'int16',
    onBuffer: (buffer) => {
      if (!buffer.data?.byteLength) return;
      setActualRate((current) => (current === buffer.sampleRate ? current : buffer.sampleRate));
      chunkRef.current(buffer.data, buffer.sampleRate);
    },
  });

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        if (enabled && !isStreaming) {
          await stream.start();
        } else if (!enabled && isStreaming) {
          stream.stop();
        }
      } catch (error) {
        if (!cancelled) onError?.(`Микрофон: ${String(error)}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [enabled, isStreaming, stream, onError]);

  return { isStreaming, actualRate };
}
