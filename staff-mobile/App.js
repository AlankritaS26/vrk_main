import React, { useEffect, useRef, useState } from 'react';
import { Alert, FlatList, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import {
  RecordingPresets,
  createAudioPlayer,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
} from 'expo-audio';
import { File, UploadType, Paths } from 'expo-file-system';
import { decode as decodeBase64, encode as base64 } from 'base-64';
import { StatusBar } from 'expo-status-bar';

const DEFAULT_BACKEND = 'http://192.168.0.117:8001';

export default function App() {
  const [backend, setBackend] = useState(DEFAULT_BACKEND);
  const [username, setUsername] = useState('staff');
  const [password, setPassword] = useState('');
  const [connected, setConnected] = useState(false);
  const [queue, setQueue] = useState([]);
  const [history, setHistory] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [micOn, setMicOn] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [status, setStatus] = useState('Enter staff credentials and connect.');
  const micRef = useRef(false);
  const recordingRef = useRef(null);
  const timerRef = useRef(null);
  const transcriptRefs = useRef({});
  const visitorPlayerRef = useRef(null);
  const visitorAudioFileRef = useRef(null);
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY, (recordingStatus) => {
    if (recordingStatus?.isRecording && micRef.current) {
      setStatus('Listening… speak normally.');
    }
  });

  const headers = () => ({
    Authorization: `Basic ${base64(`${username}:${password}`)}`,
  });

  const loadQueue = async () => {
    const response = await fetch(`${backend}/escalation/active`, { headers: headers() });
    if (!response.ok) throw new Error('Dashboard authentication failed.');
    setQueue((await response.json()).escalations || []);
  };

  const loadHistory = async () => {
    const response = await fetch(`${backend}/escalation/history?limit=50`, { headers: headers() });
    if (response.ok) setHistory((await response.json()).escalations || []);
  };

  const connect = async () => {
    if (connecting) return;
    setConnecting(true);
    setStatus('Connecting to the staff dashboard…');
    try {
      await loadQueue();
      setConnected(true);
      setStatus('Connected. Accept an escalation to speak with a visitor.');
      loadHistory().catch(() => {});
    } catch (error) {
      setStatus(error.message || 'Could not connect to the staff dashboard.');
    } finally {
      setConnecting(false);
    }
  };

  useEffect(() => {
    if (!connected) return undefined;
    const wsUrl = `${backend.replace(/^http/, 'ws')}/ws`;
    const ws = new WebSocket(wsUrl);
    ws.onmessage = async event => {
      try {
        const message = JSON.parse(event.data);
        if (message.type !== 'escalation_audio' ||
            message.speaker !== 'visitor' ||
            message.session_id !== selectedId ||
            !message.audio) return;
        const binary = decodeBase64(message.audio);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const file = new File(Paths.cache, `visitor-${Date.now()}.wav`);
        file.write(bytes);
        visitorAudioFileRef.current = file;
        if (visitorPlayerRef.current) visitorPlayerRef.current.release();
        visitorPlayerRef.current = createAudioPlayer(file.uri);
        await setAudioModeAsync({ playsInSilentMode: true });
        visitorPlayerRef.current.play();
        setStatus('Visitor is speaking…');
      } catch (error) {
        setStatus(`Visitor audio failed: ${error.message}`);
      }
    };
    ws.onerror = () => setStatus('Live visitor audio connection failed.');
    return () => {
      ws.close();
      visitorPlayerRef.current?.release();
      visitorPlayerRef.current = null;
    };
  }, [connected, backend, selectedId]);

  useEffect(() => {
    if (!connected) return undefined;
    const poll = setInterval(() => {
      loadQueue().catch(() => setStatus('Connection lost.'));
      loadHistory().catch(() => {});
    }, 4000);
    return () => clearInterval(poll);
  }, [connected, backend]);

  const accept = async (sessionId) => {
    const response = await fetch(`${backend}/escalation/accept/${sessionId}`, {
      method: 'POST', headers: headers(),
    });
    if (!response.ok) return Alert.alert('Accept failed', await response.text());
    setSelectedId(sessionId);
    await loadQueue();
  };

  const resolve = async (sessionId) => {
    await fetch(`${backend}/escalation/resolve/${sessionId}`, {
      method: 'POST', headers: headers(),
    });
    if (selectedId === sessionId) await stopMic();
    setSelectedId(null);
    await loadQueue();
    await loadHistory();
  };

  const recordClip = async (sessionId) => {
    if (!micRef.current) return;
    recordingRef.current = recorder;
    await recorder.prepareToRecordAsync();
    recorder.record();
    setStatus('Listening… speak normally for five seconds.');
    timerRef.current = setTimeout(async () => {
      try {
        await recorder.stop();
        const uri = recorder.uri;
        recordingRef.current = null;
        if (!uri) {
          throw new Error('The phone did not create an audio file.');
        }
        if (uri && micRef.current) {
          setStatus('Sending your voice…');
          let response;
          try {
            response = await new File(uri).upload(`${backend}/escalation/audio/${sessionId}`, {
              httpMethod: 'POST',
              uploadType: UploadType.MULTIPART,
              fieldName: 'audio',
              mimeType: 'audio/mp4',
              headers: headers(),
            });
          } catch (networkError) {
            throw new Error(`Cannot reach backend at ${backend}: ${networkError.message}`);
          }
          let result = {};
          try { result = response.body ? JSON.parse(response.body) : {}; } catch (_) {}
          if (response.status < 200 || response.status >= 300) {
            throw new Error(result.detail || response.body || `Voice upload failed (${response.status})`);
          }
          setStatus(result.text ? `Sent: “${result.text}”` : 'No speech detected. Please speak closer to the phone.');
        }
      } catch (error) {
        setStatus(`Voice upload failed: ${error.message}`);
      } finally {
        if (micRef.current) {
          timerRef.current = setTimeout(() => recordClip(sessionId), 250);
        }
      }
    }, 5000);
  };

  const stopMic = async () => {
    micRef.current = false;
    clearTimeout(timerRef.current);
    if (recordingRef.current) {
      try { await recordingRef.current.stop(); } catch (_) {}
      recordingRef.current = null;
    }
    setMicOn(false);
    setStatus('Microphone off.');
  };

  const toggleMic = async (sessionId) => {
    if (micRef.current) return stopMic();
    const permission = await requestRecordingPermissionsAsync();
    if (!permission.granted) {
      setStatus('Microphone permission is required for staff voice.');
      return;
    }
    try {
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
    } catch (error) {
      setStatus(`Audio setup failed: ${error.message}`);
      return;
    }
    micRef.current = true;
    setMicOn(true);
    recordClip(sessionId).catch((error) => {
      micRef.current = false;
      setMicOn(false);
      setStatus(`Recording could not start: ${error.message}`);
    });
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />
      <FlatList
        contentContainerStyle={styles.content}
        ListHeaderComponent={<View>
          <Text style={styles.title}>RNSIT Staff</Text>
          <Text style={styles.status}>{status}</Text>
          {!connected && <View style={styles.login}>
            <TextInput style={styles.input} value={backend} onChangeText={setBackend} placeholder="Backend URL" placeholderTextColor="#94a3b8" />
            <TextInput style={styles.input} value={username} onChangeText={setUsername} placeholder="Username" placeholderTextColor="#94a3b8" />
            <TextInput style={styles.input} value={password} onChangeText={setPassword} placeholder="Password" placeholderTextColor="#94a3b8" secureTextEntry />
            <Pressable style={styles.primary} onPress={connect} disabled={connecting}>
              <Text style={styles.buttonText}>{connecting ? 'Connecting…' : 'Connect'}</Text>
            </Pressable>
          </View>}
          <Text style={styles.heading}>Active escalations ({queue.length})</Text>
        </View>}
        data={queue}
        keyExtractor={item => item.session_id}
        renderItem={({ item }) => (
          <View style={styles.card}>
            <Text style={styles.name}>{item.user_name || 'Guest'}</Text>
            <Text style={styles.meta}>{item.status} · {item.reason}</Text>
            <Text style={styles.summary}>{item.summary || 'No summary available.'}</Text>
            <ScrollView
              ref={ref => { transcriptRefs.current[item.session_id] = ref; }}
              style={styles.transcriptScroll}
              contentContainerStyle={styles.transcriptContent}
              nestedScrollEnabled
              onContentSizeChange={() => transcriptRefs.current[item.session_id]?.scrollToEnd({ animated: true })}
            >
              <Text style={styles.transcript}>{(item.transcript || []).slice(-20).map(m => `${m.speaker}: ${m.text}`).join('\n')}</Text>
            </ScrollView>
            {item.status === 'STAFF_NOTIFIED' && <Pressable style={styles.primary} onPress={() => accept(item.session_id)}><Text style={styles.buttonText}>Accept</Text></Pressable>}
            {item.status === 'STAFF_CONNECTED' && <><Pressable style={micOn && selectedId === item.session_id ? styles.danger : styles.voice} onPress={() => { setSelectedId(item.session_id); toggleMic(item.session_id); }}><Text style={styles.buttonText}>{micOn && selectedId === item.session_id ? 'Turn microphone off' : 'Turn microphone on'}</Text></Pressable><Pressable style={styles.resolve} onPress={() => resolve(item.session_id)}><Text style={styles.buttonText}>Resolve</Text></Pressable></>}
          </View>
        )}
        ListFooterComponent={<View><Text style={styles.heading}>Recent history</Text>{history.map(item => <View style={styles.history} key={`${item.session_id}-${item.created_at}`}><Text style={styles.name}>{item.user_name || 'Guest'} · {item.status}</Text><Text style={styles.meta}>{item.summary || 'No summary available.'}</Text><Text style={styles.transcript}>{(item.transcript || []).map(m => `${m.speaker}: ${m.text}`).join('\n')}</Text></View>)}</View>}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#0f172a' },
  content: { padding: 16 },
  title: { color: '#f8fafc', fontSize: 28, fontWeight: '800', marginTop: 18 },
  status: { color: '#93c5fd', marginVertical: 8 },
  login: { gap: 10, marginBottom: 18 },
  input: { backgroundColor: '#1e293b', color: '#f8fafc', borderRadius: 10, padding: 12 },
  heading: { color: '#cbd5e1', fontSize: 18, fontWeight: '700', marginVertical: 14 },
  card: { backgroundColor: '#1e293b', borderRadius: 14, padding: 16, marginBottom: 12 },
  history: { backgroundColor: '#172033', borderRadius: 12, padding: 14, marginBottom: 10 },
  name: { color: '#f8fafc', fontSize: 17, fontWeight: '700' },
  meta: { color: '#94a3b8', marginTop: 4 },
  summary: { color: '#cbd5e1', marginTop: 10 },
  transcript: { color: '#bfdbfe', backgroundColor: '#0b1220', padding: 10, marginVertical: 10, borderRadius: 8, lineHeight: 20 },
  transcriptScroll: { maxHeight: 190, marginVertical: 10, borderRadius: 8 },
  transcriptContent: { flexGrow: 1 },
  primary: { backgroundColor: '#2563eb', borderRadius: 10, padding: 13, alignItems: 'center', marginTop: 8 },
  voice: { backgroundColor: '#059669', borderRadius: 10, padding: 13, alignItems: 'center', marginTop: 8 },
  danger: { backgroundColor: '#dc2626', borderRadius: 10, padding: 13, alignItems: 'center', marginTop: 8 },
  resolve: { backgroundColor: '#6366f1', borderRadius: 10, padding: 13, alignItems: 'center', marginTop: 8 },
  buttonText: { color: '#fff', fontWeight: '700' },
});
