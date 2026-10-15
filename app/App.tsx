import { NavigationContainer, useNavigation } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { RecordingPresets, requestRecordingPermissionsAsync, setAudioModeAsync, useAudioRecorder } from 'expo-audio';
import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { Pressable, SafeAreaView, StyleSheet, Text, View } from 'react-native';

type RootStackParamList = {
  Home: undefined;
  Device: undefined;
  Routines: undefined;
  Confirm: { action?: string } | undefined;
};

type Command = 'yes' | 'no' | 'up' | 'down' | 'left' | 'right' | 'on' | 'off' | 'go' | 'stop';

const Stack = createNativeStackNavigator<RootStackParamList>();

const commandLabels: Record<Command, string> = {
  yes: 'YES', no: 'NO', up: 'UP', down: 'DOWN', left: 'LEFT', right: 'RIGHT',
  on: 'ON', off: 'OFF', go: 'GO', stop: 'STOP',
};

function AppShell({ children, title, eyebrow }: { children: React.ReactNode; title: string; eyebrow: string }) {
  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />
      <View style={styles.topbar}>
        <View style={styles.statusDot} />
        <Text style={styles.brand}>VOICE / HOME</Text>
        <Text style={styles.online}>LOCAL</Text>
      </View>
      <View style={styles.content}>
        <Text style={styles.eyebrow}>{eyebrow}</Text>
        <Text style={styles.title}>{title}</Text>
        {children}
      </View>
    </SafeAreaView>
  );
}

function VoiceControl({ onCommand }: { onCommand: (command: Command) => void }) {
  const recorder = useAudioRecorder(RecordingPresets.LOW_QUALITY);
  const [recording, setRecording] = useState(false);
  const [message, setMessage] = useState('Ready for a command');

  const recordWindow = async () => {
    if (recording) return;
    const permission = await requestRecordingPermissionsAsync();
    if (!permission.granted) {
      setMessage('Microphone permission required');
      return;
    }
    await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
    await recorder.prepareToRecordAsync();
    recorder.record();
    setRecording(true);
    setMessage('Listening for 1 second...');
    setTimeout(async () => {
      recorder.stop();
      setRecording(false);
      setMessage('Audio captured. ONNX inference is next.');
    }, 1000);
  };

  return (
    <View style={styles.voicePanel}>
      <View>
        <Text style={styles.panelLabel}>VOICE INPUT</Text>
        <Text style={styles.voiceMessage}>{message}</Text>
      </View>
      <Pressable style={[styles.micButton, recording && styles.micButtonActive]} onPress={() => void recordWindow()}>
        <Text style={styles.micIcon}>{recording ? '■' : '●'}</Text>
      </Pressable>
    </View>
  );
}

function HomeScreen() {
  const navigation = useNavigation<any>();
  return (
    <AppShell eyebrow="CONTROL CENTER" title="Smart Home.">
      <Text style={styles.subtitle}>Speak a command.</Text>
      <View style={styles.grid}>
        <NavCard title="Living room" detail="2 devices online" accent="#d7ff64" onPress={() => navigation.navigate('Device')} />
        <NavCard title="Routines" detail="3 saved scenes" accent="#65d9ff" onPress={() => navigation.navigate('Routines')} />
      </View>
      <VoiceControl onCommand={(command) => { if (command === 'go' || command === 'yes') navigation.navigate('Device'); }} />
      <View style={styles.quickRow}>
        <QuickCommand command="up" /><QuickCommand command="down" /><QuickCommand command="left" /><QuickCommand command="right" /><QuickCommand command="go" />
      </View>
      <Text style={styles.hint}>Use GO to open the focused room.</Text>
    </AppShell>
  );
}

function DeviceScreen() {
  const navigation = useNavigation<any>();
  const [powered, setPowered] = useState(true);
  return (
    <AppShell eyebrow="LIVING ROOM" title="Main lights">
      <View style={styles.deviceHero}>
        <Text style={styles.deviceGlyph}>◉</Text>
        <View><Text style={styles.deviceName}>Ceiling lamp</Text><Text style={styles.deviceMeta}>Living room · Smart relay</Text></View>
        <View style={[styles.powerPill, powered ? styles.powerOn : styles.powerOff]}><Text style={styles.powerText}>{powered ? 'ON' : 'OFF'}</Text></View>
      </View>
      <VoiceControl onCommand={(command) => { if (command === 'on') setPowered(true); if (command === 'off') setPowered(false); if (command === 'go') setPowered(true); if (command === 'stop' || command === 'no') navigation.goBack(); }} />
      <View style={styles.quickRow}>
        <QuickCommand command="on" /><QuickCommand command="off" /><QuickCommand command="go" /><QuickCommand command="stop" />
      </View>
      <Text style={styles.hint}>Say ON or OFF to control the device, or STOP to return.</Text>
    </AppShell>
  );
}

function RoutinesScreen() {
  const navigation = useNavigation<any>();
  return (
    <AppShell eyebrow="AUTOMATIONS" title="Your routines">
      <RoutineCard name="Welcome home" detail="Lights on · 18:30" onPress={() => navigation.navigate('Confirm', { action: 'Welcome home' })} />
      <RoutineCard name="Movie mode" detail="Lights low · TV on" onPress={() => navigation.navigate('Confirm', { action: 'Movie mode' })} />
      <RoutineCard name="Good night" detail="Everything off · 23:00" onPress={() => navigation.navigate('Confirm', { action: 'Good night' })} />
      <VoiceControl onCommand={(command) => { if (command === 'go') navigation.navigate('Confirm', { action: 'Welcome home' }); if (command === 'stop') navigation.goBack(); }} />
      <View style={styles.quickRow}><QuickCommand command="go" /><QuickCommand command="stop" /></View>
      <Text style={styles.hint}>Use GO to start the selected routine or STOP to leave.</Text>
    </AppShell>
  );
}

function ConfirmScreen({ route }: { route: { params?: { action?: string } } }) {
  const navigation = useNavigation<any>();
  const action = route.params?.action ?? 'Welcome home';
  return (
    <AppShell eyebrow="CONFIRMATION" title="Ready to run?">
      <View style={styles.confirmCard}><Text style={styles.confirmMark}>✓</Text><Text style={styles.confirmTitle}>{action}</Text><Text style={styles.confirmDetail}>This routine will update your living room devices.</Text></View>
      <VoiceControl onCommand={(command) => { if (command === 'go' || command === 'yes') navigation.navigate('Home'); if (command === 'stop' || command === 'no') navigation.goBack(); }} />
      <View style={styles.quickRow}><QuickCommand command="go" /><QuickCommand command="stop" /></View>
      <Text style={styles.hint}>Use GO to confirm the routine or STOP to cancel.</Text>
    </AppShell>
  );
}

function NavCard({ title, detail, accent, onPress }: { title: string; detail: string; accent: string; onPress: () => void }) {
  return <Pressable style={styles.navCard} onPress={onPress}><View style={[styles.cardStripe, { backgroundColor: accent }]} /><Text style={styles.cardTitle}>{title}</Text><Text style={styles.cardDetail}>{detail}</Text><Text style={styles.cardArrow}>↗</Text></Pressable>;
}

function RoutineCard({ name, detail, onPress }: { name: string; detail: string; onPress: () => void }) {
  return <Pressable style={styles.routineCard} onPress={onPress}><View><Text style={styles.cardTitle}>{name}</Text><Text style={styles.cardDetail}>{detail}</Text></View><Text style={styles.cardArrow}>→</Text></Pressable>;
}

function QuickCommand({ command }: { command: Command }) { return <View style={styles.quickCommand}><Text style={styles.quickText}>{commandLabels[command]}</Text></View>; }

export default function App() {
  return <NavigationContainer><Stack.Navigator screenOptions={{ headerShown: false }}><Stack.Screen name="Home" component={HomeScreen} /><Stack.Screen name="Device" component={DeviceScreen} /><Stack.Screen name="Routines" component={RoutinesScreen} /><Stack.Screen name="Confirm" component={ConfirmScreen} /></Stack.Navigator></NavigationContainer>;
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#101312' },
  topbar: { height: 64, paddingHorizontal: 22, flexDirection: 'row', alignItems: 'center', borderBottomWidth: 1, borderBottomColor: '#29302c' },
  statusDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#d7ff64', marginRight: 10 },
  brand: { color: '#f2f3ed', fontSize: 12, fontWeight: '800', letterSpacing: 2 },
  online: { marginLeft: 'auto', color: '#87918a', fontSize: 11, letterSpacing: 1.5 },
  content: { flex: 1, padding: 24 },
  eyebrow: { color: '#87918a', fontSize: 11, fontWeight: '800', letterSpacing: 2, marginTop: 14 },
  title: { color: '#f2f3ed', fontSize: 32, fontWeight: '800', marginTop: 10, letterSpacing: -0.5 },
  subtitle: { color: '#9ca69f', fontSize: 15, marginTop: 10, marginBottom: 24 },
  grid: { flexDirection: 'row', gap: 12 },
  navCard: { flex: 1, minHeight: 142, backgroundColor: '#1a201e', padding: 18, borderRadius: 6, overflow: 'hidden' },
  cardStripe: { width: 32, height: 4, marginBottom: 26 },
  cardTitle: { color: '#f2f3ed', fontSize: 17, fontWeight: '700' },
  cardDetail: { color: '#89938c', fontSize: 12, marginTop: 7 },
  cardArrow: { color: '#d7ff64', fontSize: 22, position: 'absolute', right: 16, bottom: 14 },
  voicePanel: { marginTop: 26, padding: 18, borderWidth: 1, borderColor: '#39433d', borderRadius: 6, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  panelLabel: { color: '#d7ff64', fontSize: 10, fontWeight: '800', letterSpacing: 2 },
  voiceMessage: { color: '#c8cec8', marginTop: 8, fontSize: 13 },
  micButton: { width: 54, height: 54, borderRadius: 27, backgroundColor: '#d7ff64', alignItems: 'center', justifyContent: 'center' },
  micButtonActive: { backgroundColor: '#ff765f' },
  micIcon: { color: '#101312', fontSize: 18 },
  quickRow: { flexDirection: 'row', gap: 10, marginTop: 18 },
  quickCommand: { flex: 1, height: 42, borderRadius: 5, backgroundColor: '#1a201e', alignItems: 'center', justifyContent: 'center' },
  quickText: { color: '#87918a', fontSize: 10, fontWeight: '800', letterSpacing: 1 },
  deviceHero: { backgroundColor: '#1a201e', marginTop: 28, padding: 20, borderRadius: 6, flexDirection: 'row', alignItems: 'center' },
  deviceGlyph: { color: '#d7ff64', fontSize: 34, marginRight: 16 },
  deviceName: { color: '#f2f3ed', fontSize: 18, fontWeight: '700' },
  deviceMeta: { color: '#89938c', fontSize: 12, marginTop: 5 },
  powerPill: { marginLeft: 'auto', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 12 },
  powerOn: { backgroundColor: '#d7ff64' }, powerOff: { backgroundColor: '#39433d' },
  powerText: { color: '#101312', fontSize: 10, fontWeight: '900' },
  textButton: { marginTop: 24, alignSelf: 'flex-start' },
  textButtonLabel: { color: '#a9b1aa', fontSize: 13 },
  routineCard: { marginTop: 14, padding: 18, backgroundColor: '#1a201e', borderRadius: 6, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  hint: { color: '#7f8982', fontSize: 12, marginTop: 16 },
  confirmCard: { backgroundColor: '#1a201e', marginTop: 28, padding: 26, borderRadius: 6, alignItems: 'center' },
  confirmMark: { color: '#d7ff64', fontSize: 42 },
  confirmTitle: { color: '#f2f3ed', fontSize: 22, fontWeight: '800', marginTop: 16 },
  confirmDetail: { color: '#89938c', fontSize: 13, textAlign: 'center', marginTop: 10, lineHeight: 20 },
});
