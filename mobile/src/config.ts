/**
 * Настройки приложения: адрес голосового сервиса и выбранный голос.
 * Хранятся на устройстве, чтобы не вводить адрес при каждом запуске.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

export interface AppConfig {
  /** Полный адрес WebSocket, например ws://100.79.158.109:8643/v1/voice */
  serverUrl: string;
  /**
   * Выбранный голос. Пустая строка — «не выбирал»: тогда клиент возьмёт
   * серверный по умолчанию. Хардкодить имя нельзя: у edge-tts и Piper разные
   * пространства имён, и имя от одного движка сервер с другим отвергнет.
   */
  voice: string;
}

export const DEFAULT_CONFIG: AppConfig = {
  // Tailscale-адрес мини-ПК, где работает hermes-voice.
  serverUrl: 'ws://100.79.158.109:8643/v1/voice',
  voice: '',
};

const STORAGE_KEY = 'hermes-voice.config';

export async function loadConfig(): Promise<AppConfig> {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_CONFIG;
    const parsed = JSON.parse(raw) as Partial<AppConfig>;
    return { ...DEFAULT_CONFIG, ...parsed };
  } catch {
    return DEFAULT_CONFIG;
  }
}

export async function saveConfig(config: AppConfig): Promise<void> {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

/** Разобрать адрес в удобные части: ws://host:port/v1/voice */
export function parseServerUrl(url: string): { host: string; port: string; path: string } {
  const match = /^wss?:\/\/([^:/]+)(?::(\d+))?(\/.*)?$/.exec(url.trim());
  if (!match) return { host: '', port: '8643', path: '/v1/voice' };
  return {
    host: match[1],
    port: match[2] ?? (url.startsWith('wss') ? '443' : '8643'),
    path: match[3] ?? '/v1/voice',
  };
}

/** Собрать адрес обратно из частей. */
export function buildServerUrl(host: string, port: string): string {
  return `ws://${host.trim()}:${port.trim() || '8643'}/v1/voice`;
}
