/**
 * Единственный экран: разговор с агентом.
 *
 * Логика намеренно простая — одна большая кнопка и то, что происходит в диалоге.
 * Ничего не нужно нажимать между репликами: сервер сам понимает, когда человек
 * закончил говорить.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Animated,
  Easing,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';

import { buildServerUrl, loadConfig, parseServerUrl, saveConfig, type AppConfig } from '../config';
import { requestMicPermission, useMicrophone } from '../audio/recorder';
import { VoiceSession, type SessionSnapshot } from '../voice/VoiceSession';
import type { Backend, VoiceState } from '../voice/protocol';

const COLORS = {
  background: '#0F1115',
  surface: '#171A21',
  border: '#252A34',
  text: '#E8EAED',
  muted: '#8A9199',
  accent: '#4C8DFF',
  accentDim: '#1E3A66',
  danger: '#E5484D',
  ok: '#3DD68C',
};

const STATE_LABEL: Record<VoiceState, string> = {
  idle: 'Готов',
  listening: 'Слушаю…',
  thinking: 'Думаю…',
  speaking: 'Говорю',
};

/**
 * Режим ответа. Разница не косметическая:
 *  direct — модель отвечает сама, быстро, но без инструментов;
 *  agent  — полный Hermes: поиск в вебе, файлы, память, но в разы дольше.
 */
const BACKEND_LABEL: Record<Backend, string> = {
  direct: 'Быстрый',
  agent: 'Агент',
};

/** Человеческое имя голоса из технического (ru_RU-irina-medium → Irina). */
function voiceLabel(name: string): string {
  const bare = name.replace(/^ru[-_]RU[-_]/, '');
  const word = bare.split(/[-_]/)[0] ?? name;
  return word.charAt(0).toUpperCase() + word.slice(1);
}

export default function VoiceScreen() {
  const session = useMemo(() => new VoiceSession(), []);
  const [snapshot, setSnapshot] = useState<SessionSnapshot>(session.getSnapshot());
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [draftHost, setDraftHost] = useState('');
  const [draftPort, setDraftPort] = useState('8643');
  const [textInput, setTextInput] = useState('');
  const [textMode, setTextMode] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);

  const pulse = useRef(new Animated.Value(0)).current;
  const scrollRef = useRef<ScrollView>(null);

  // --- настройки при старте ---
  useEffect(() => {
    void (async () => {
      const stored = await loadConfig();
      setConfig(stored);
      const { host, port } = parseServerUrl(stored.serverUrl);
      setDraftHost(host);
      setDraftPort(port);
      await session.connect(stored.serverUrl, stored.voice);
    })();
    return () => {
      void session.disconnect();
    };
  }, [session]);

  // --- подписка на состояние диалога ---
  useEffect(() => session.subscribe(setSnapshot), [session]);

  // --- сервер поправил голос (сохранённый остался от другого движка) — запоминаем ---
  useEffect(() => {
    if (!snapshot.ready || !snapshot.voice) return;
    if (config?.voice === snapshot.voice) return;
    const next: AppConfig = {
      ...(config ?? { serverUrl: '', voice: snapshot.voice }),
      voice: snapshot.voice,
    };
    setConfig(next);
    void saveConfig(next);
  }, [snapshot.ready, snapshot.voice, config]);

  // --- микрофон: потоком управляет состояние диалога ---
  const { isStreaming: micLive } = useMicrophone({
    enabled: snapshot.micWanted && snapshot.connected,
    onChunk: (pcm, rate) => session.pushAudio(pcm, rate),
    onError: (message) => setMicError(message),
  });

  // --- пульсация кнопки по громкости речи ---
  useEffect(() => {
    const active = snapshot.state === 'listening';
    Animated.timing(pulse, {
      toValue: active ? Math.min(0.35 + snapshot.level * 0.65, 1) : 0,
      duration: 120,
      easing: Easing.out(Easing.quad),
      useNativeDriver: true,
    }).start();
  }, [snapshot.level, snapshot.state, pulse]);

  useEffect(() => {
    const timer = setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 80);
    return () => clearTimeout(timer);
  }, [snapshot.answer, snapshot.transcript]);

  const busy = snapshot.state === 'thinking' || snapshot.state === 'speaking';
  const listening = snapshot.state === 'listening';

  const toggleListening = async () => {
    setMicError(null);
    if (listening) {
      await session.stopListening();
      return;
    }
    const granted = await requestMicPermission();
    if (!granted) {
      setMicError('Нет доступа к микрофону');
      return;
    }
    if (busy) await session.interrupt();
    await session.startListening();
  };

  const applySettings = async () => {
    const url = buildServerUrl(draftHost, draftPort);
    const next: AppConfig = { ...(config ?? { serverUrl: url, voice: '' }), serverUrl: url };
    setConfig(next);
    await saveConfig(next);
    setSettingsOpen(false);
    await session.disconnect();
    await session.connect(next.serverUrl, next.voice);
  };

  /** Выбор голоса: применяется сразу, без переподключения. */
  const pickVoice = async (voice: string) => {
    session.setVoice(voice);
    const next: AppConfig = { ...(config ?? { serverUrl: '', voice }), voice };
    setConfig(next);
    await saveConfig(next);
  };

  return (
    <SafeAreaView style={styles.root}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        {/* Шапка */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <View
              style={[
                styles.dot,
                { backgroundColor: snapshot.connected ? COLORS.ok : COLORS.danger },
              ]}
            />
            <Text style={styles.headerTitle}>Hermes</Text>
            {/* Режим виден всегда: иначе непонятно, почему ответ идёт то быстро, то долго */}
            <Text style={styles.headerBadge}>
              {BACKEND_LABEL[snapshot.backend] ?? snapshot.backend}
            </Text>
          </View>
          <Pressable onPress={() => setSettingsOpen((value) => !value)} hitSlop={12}>
            <Text style={styles.headerAction}>{settingsOpen ? 'Готово' : 'Настройки'}</Text>
          </Pressable>
        </View>

        {settingsOpen && (
          <View style={styles.settings}>
            <Text style={styles.label}>Адрес сервера</Text>
            <View style={styles.row}>
              <TextInput
                style={[styles.input, styles.flex]}
                value={draftHost}
                onChangeText={setDraftHost}
                placeholder="100.79.158.109"
                placeholderTextColor={COLORS.muted}
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="numbers-and-punctuation"
              />
              <TextInput
                style={[styles.input, styles.portInput]}
                value={draftPort}
                onChangeText={setDraftPort}
                placeholder="8643"
                placeholderTextColor={COLORS.muted}
                keyboardType="number-pad"
              />
            </View>
            <View style={styles.switchRow}>
              <Text style={styles.label}>Писать текстом</Text>
              <Switch value={textMode} onValueChange={setTextMode} />
            </View>

            {/* Режим ответа: скорость против инструментов */}
            {snapshot.backends.length > 1 && (
              <>
                <Text style={styles.label}>Режим ответа</Text>
                <View style={styles.segment}>
                  {snapshot.backends.map((item) => (
                    <Pressable
                      key={item}
                      style={[styles.segmentItem, snapshot.backend === item && styles.segmentItemActive]}
                      onPress={() => session.setBackend(item)}
                    >
                      <Text
                        style={[
                          styles.segmentText,
                          snapshot.backend === item && styles.segmentTextActive,
                        ]}
                      >
                        {BACKEND_LABEL[item] ?? item}
                      </Text>
                    </Pressable>
                  ))}
                </View>
                <Text style={styles.note}>
                  {snapshot.backend === 'agent'
                    ? 'Агент: умеет искать в вебе и читать файлы. Отвечает дольше.'
                    : 'Быстрый: отвечает за пару секунд, но без инструментов.'}
                </Text>
              </>
            )}

            {/* Голос: список приходит с сервера */}
            {snapshot.voices.length > 0 && (
              <>
                <Text style={styles.label}>Голос</Text>
                <View style={styles.chips}>
                  {snapshot.voices.map((item) => (
                    <Pressable
                      key={item}
                      style={[styles.chip, snapshot.voice === item && styles.chipActive]}
                      onPress={() => void pickVoice(item)}
                    >
                      <Text
                        style={[styles.chipText, snapshot.voice === item && styles.chipTextActive]}
                      >
                        {voiceLabel(item)}
                      </Text>
                    </Pressable>
                  ))}
                </View>
              </>
            )}

            <Pressable style={styles.primaryButton} onPress={applySettings}>
              <Text style={styles.primaryButtonText}>Переподключиться</Text>
            </Pressable>
          </View>
        )}

        {/* Диалог */}
        <ScrollView ref={scrollRef} style={styles.flex} contentContainerStyle={styles.dialog}>
          {snapshot.transcript ? (
            <View style={[styles.bubble, styles.userBubble]}>
              <Text style={styles.bubbleLabel}>Вы</Text>
              <Text style={styles.bubbleText}>{snapshot.transcript}</Text>
            </View>
          ) : null}

          {snapshot.answer ? (
            <View style={[styles.bubble, styles.agentBubble]}>
              <Text style={styles.bubbleLabel}>Hermes</Text>
              <Text style={styles.bubbleText}>{snapshot.answer}</Text>
            </View>
          ) : null}

          {!snapshot.transcript && !snapshot.answer ? (
            <Text style={styles.hint}>
              {snapshot.ready
                ? 'Нажмите кнопку и говорите — агент сам поймёт, когда вы закончили.'
                : 'Подключение к серверу…'}
            </Text>
          ) : null}

          {snapshot.error ? <Text style={styles.error}>{snapshot.error}</Text> : null}
          {micError ? <Text style={styles.error}>{micError}</Text> : null}
        </ScrollView>

        {/* Поле ввода текста */}
        {textMode && (
          <View style={styles.textRow}>
            <TextInput
              style={[styles.input, styles.flex]}
              value={textInput}
              onChangeText={setTextInput}
              placeholder="Сообщение…"
              placeholderTextColor={COLORS.muted}
              onSubmitEditing={() => {
                const value = textInput.trim();
                if (!value) return;
                session.sendText(value);
                setTextInput('');
              }}
              returnKeyType="send"
            />
            <Pressable
              style={styles.sendButton}
              onPress={() => {
                const value = textInput.trim();
                if (!value) return;
                session.sendText(value);
                setTextInput('');
              }}
            >
              <Text style={styles.sendButtonText}>→</Text>
            </Pressable>
          </View>
        )}

        {/* Нижняя панель */}
        <View style={styles.controls}>
          <Text style={styles.state}>{STATE_LABEL[snapshot.state]}</Text>

          <View style={styles.buttonWrap}>
            <Animated.View
              pointerEvents="none"
              style={[
                styles.halo,
                {
                  opacity: pulse,
                  transform: [{ scale: pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.35] }) }],
                },
              ]}
            />
            <Pressable
              onPress={toggleListening}
              onLongPress={() => void session.interrupt()}
              style={[
                styles.micButton,
                listening && styles.micButtonActive,
                busy && styles.micButtonBusy,
              ]}
            >
              {snapshot.state === 'thinking' ? (
                <ActivityIndicator color={COLORS.text} />
              ) : (
                <Text style={styles.micIcon}>{listening ? '■' : '●'}</Text>
              )}
            </Pressable>
          </View>

          <Text style={styles.hintSmall}>
            {listening
              ? micLive
                ? `Нажмите, чтобы остановить${snapshot.micRate ? ` · ${snapshot.micRate} Гц` : ''}`
                : 'Включаю микрофон…'
              : 'Нажмите и говорите'}
          </Text>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.background },
  flex: { flex: 1 },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: COLORS.border,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  headerTitle: { color: COLORS.text, fontSize: 17, fontWeight: '500' },
  headerBadge: {
    color: COLORS.muted,
    fontSize: 11,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: COLORS.border,
    borderRadius: 8,
    paddingHorizontal: 7,
    paddingVertical: 2,
  },
  headerAction: { color: COLORS.accent, fontSize: 14 },
  dot: { width: 8, height: 8, borderRadius: 4 },

  settings: {
    padding: 16,
    gap: 12,
    backgroundColor: COLORS.surface,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: COLORS.border,
  },
  label: { color: COLORS.muted, fontSize: 13 },
  note: { color: COLORS.muted, fontSize: 12, lineHeight: 17, marginTop: -6 },
  row: { flexDirection: 'row', gap: 8 },
  switchRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },

  segment: {
    flexDirection: 'row',
    backgroundColor: COLORS.background,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: COLORS.border,
    padding: 3,
    gap: 3,
  },
  segmentItem: { flex: 1, paddingVertical: 9, alignItems: 'center', borderRadius: 8 },
  segmentItemActive: { backgroundColor: COLORS.accentDim },
  segmentText: { color: COLORS.muted, fontSize: 14 },
  segmentTextActive: { color: COLORS.text },

  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 16,
    backgroundColor: COLORS.background,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: COLORS.border,
  },
  chipActive: { backgroundColor: COLORS.accentDim, borderColor: COLORS.accent },
  chipText: { color: COLORS.muted, fontSize: 14 },
  chipTextActive: { color: COLORS.text },
  input: {
    backgroundColor: COLORS.background,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: COLORS.border,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: COLORS.text,
    fontSize: 15,
  },
  portInput: { width: 84 },
  primaryButton: {
    backgroundColor: COLORS.accentDim,
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
  },
  primaryButtonText: { color: COLORS.text, fontSize: 15 },

  dialog: { padding: 16, gap: 12, flexGrow: 1 },
  bubble: { borderRadius: 14, padding: 12, maxWidth: '92%' },
  userBubble: { alignSelf: 'flex-end', backgroundColor: COLORS.accentDim },
  agentBubble: {
    alignSelf: 'flex-start',
    backgroundColor: COLORS.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: COLORS.border,
  },
  bubbleLabel: { color: COLORS.muted, fontSize: 11, marginBottom: 4 },
  bubbleText: { color: COLORS.text, fontSize: 15, lineHeight: 22 },
  hint: { color: COLORS.muted, fontSize: 14, textAlign: 'center', marginTop: 32, lineHeight: 21 },
  error: { color: COLORS.danger, fontSize: 13, textAlign: 'center' },

  textRow: { flexDirection: 'row', gap: 8, paddingHorizontal: 16, paddingBottom: 8 },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 10,
    backgroundColor: COLORS.accentDim,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendButtonText: { color: COLORS.text, fontSize: 20 },

  controls: { alignItems: 'center', paddingBottom: 24, paddingTop: 8, gap: 12 },
  state: { color: COLORS.muted, fontSize: 13 },
  buttonWrap: { width: 108, height: 108, alignItems: 'center', justifyContent: 'center' },
  halo: {
    position: 'absolute',
    width: 108,
    height: 108,
    borderRadius: 54,
    backgroundColor: COLORS.accent,
  },
  micButton: {
    width: 84,
    height: 84,
    borderRadius: 42,
    backgroundColor: COLORS.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  micButtonActive: { backgroundColor: COLORS.danger },
  micButtonBusy: { backgroundColor: COLORS.accentDim },
  micIcon: { color: COLORS.text, fontSize: 26 },
  hintSmall: { color: COLORS.muted, fontSize: 12 },
});
