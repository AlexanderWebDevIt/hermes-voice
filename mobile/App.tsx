/**
 * Hermes Voice — голосовой клиент к агенту Hermes.
 *
 * Приложение тонкое: оно отдаёт микрофон на сервер hermes-voice и проигрывает то,
 * что приходит обратно. Вся логика диалога живёт на сервере, рядом с агентом.
 */

import { StatusBar } from 'expo-status-bar';

import VoiceScreen from './src/screens/VoiceScreen';

export default function App() {
  return (
    <>
      <StatusBar style="light" />
      <VoiceScreen />
    </>
  );
}
