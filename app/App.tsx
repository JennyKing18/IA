import { NavigationContainer, useNavigation } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import { createContext, useContext, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

import { useVoiceCommand } from './src/audio/useVoiceCommand';

type DeviceId = 'lights' | 'fan' | 'tv';

type RootStackParamList = {
  Home: undefined;
  Device: { deviceId?: DeviceId } | undefined;
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

// Micrófono → log-mel → modelo ONNX en el celular → onCommand (ver src/audio/useVoiceCommand.ts)
function VoiceControl({ onCommand }: { onCommand: (command: Command) => void }) {
  const { state, message, listen } = useVoiceCommand(onCommand);
  const busy = state === 'listening' || state === 'thinking';

  return (
    <View style={styles.voicePanel}>
      <View style={styles.deviceText}>
        <Text style={styles.panelLabel}>VOICE INPUT</Text>
        <Text style={styles.voiceMessage}>{message}</Text>
      </View>
      <Pressable style={[styles.micButton, busy && styles.micButtonActive, state === 'loading' && styles.micButtonDisabled]}
        onPress={() => void listen()} disabled={state !== 'idle'}>
        <Text style={styles.micIcon}>{state === 'listening' ? '■' : state === 'thinking' ? '…' : '●'}</Text>
      </Pressable>
    </View>
  );
}

// Estado simulado de los dispositivos, compartido entre el panel principal y la pantalla de dispositivo
type DeviceInfo = { name: string; detail: string; on: boolean };
type DevicesState = { devices: Record<DeviceId, DeviceInfo>; setPower: (id: DeviceId, on: boolean) => void };

const INITIAL_DEVICES: Record<DeviceId, DeviceInfo> = {
  lights: { name: 'Lights', detail: 'Living room and bedroom lights', on: true },
  fan: { name: 'Fan', detail: 'Air circulation and comfort mode', on: false },
  tv: { name: 'TV', detail: 'Media and ambient screen', on: false },
};

const DevicesContext = createContext<DevicesState | null>(null);

function DevicesProvider({ children }: { children: React.ReactNode }) {
  const [devices, setDevices] = useState(INITIAL_DEVICES);
  const setPower = (id: DeviceId, on: boolean) => setDevices((prev) => ({ ...prev, [id]: { ...prev[id], on } }));
  return <DevicesContext.Provider value={{ devices, setPower }}>{children}</DevicesContext.Provider>;
}

function useDevices() {
  const state = useContext(DevicesContext);
  if (!state) throw new Error('useDevices must be used inside DevicesProvider');
  return state;
}

// Panel principal: cuadrícula 2 × 2. up/down/left/right mueven el foco; yes abre la tarjeta.
const HOME_CARDS: (DeviceId | 'routines')[] = ['lights', 'fan', 'tv', 'routines'];
const HOME_COLUMNS = 2;

function HomeScreen() {
  const navigation = useNavigation<any>();
  const { devices } = useDevices();
  const [focus, setFocus] = useState(0);
  const [lastCommand, setLastCommand] = useState<Command | null>(null);

  const open = (index: number) => {
    const card = HOME_CARDS[index];
    if (card === 'routines') navigation.navigate('Routines');
    else navigation.navigate('Device', { deviceId: card });
  };

  const onCommand = (command: Command) => {
    setLastCommand(command);
    const column = focus % HOME_COLUMNS;
    if (command === 'up' && focus - HOME_COLUMNS >= 0) setFocus(focus - HOME_COLUMNS);
    if (command === 'down' && focus + HOME_COLUMNS < HOME_CARDS.length) setFocus(focus + HOME_COLUMNS);
    if (command === 'left' && column > 0) setFocus(focus - 1);
    if (command === 'right' && column < HOME_COLUMNS - 1 && focus + 1 < HOME_CARDS.length) setFocus(focus + 1);
    if (command === 'yes') open(focus);
  };

  return (
    <AppShell eyebrow="CONTROL CENTER" title="Smart Home.">
      <Text style={styles.subtitle}>Listening: navigation commands</Text>
      <View style={styles.homeGrid}>
        {HOME_CARDS.map((card, index) => {
          const isRoutines = card === 'routines';
          const info = isRoutines ? { name: 'Routines', detail: 'Study, night and welcome scenes', on: false } : devices[card];
          const badge = isRoutines ? 'OPEN' : info.on ? 'ON' : 'OFF';
          return (
            <Pressable key={card} style={[styles.homeCard, focus === index && styles.focused]}
              onPress={() => { setFocus(index); open(index); }}>
              <Text style={styles.cardTitle}>{info.name}</Text>
              <Text style={styles.cardDetail}>{info.detail}</Text>
              <View style={[styles.badge, isRoutines ? styles.badgeOpen : info.on ? styles.powerOn : styles.powerOff]}>
                <Text style={[styles.powerText, !isRoutines && !info.on && styles.badgeTextOff]}>{badge}</Text>
              </View>
            </Pressable>
          );
        })}
      </View>
      <VoiceControl onCommand={onCommand} />
      <View style={styles.quickRow}>
        {(['up', 'down', 'left', 'right', 'yes'] as Command[]).map((c) => <QuickCommand key={c} command={c} onPress={() => onCommand(c)} />)}
      </View>
      <Text style={styles.hint}>
        Use UP/DOWN/LEFT/RIGHT to move the focus and YES to open the selected card.{lastCommand ? `  Last: ${commandLabels[lastCommand]}` : ''}
      </Text>
    </AppShell>
  );
}

// Dispositivo: on/off encienden o apagan; las flechas mueven el foco; yes ejecuta la acción enfocada; no regresa.
type DeviceAction = 'on' | 'off' | 'back';

function DeviceScreen({ route }: { route: { params?: { deviceId?: DeviceId } } }) {
  const navigation = useNavigation<any>();
  const deviceId = route.params?.deviceId ?? 'lights';
  const { devices, setPower } = useDevices();
  const device = devices[deviceId];
  const [focus, setFocus] = useState<DeviceAction>(device.on ? 'off' : 'on');
  const [lastCommand, setLastCommand] = useState<Command | null>(null);

  const run = (action: DeviceAction) => {
    if (action === 'on') setPower(deviceId, true);
    if (action === 'off') setPower(deviceId, false);
    if (action === 'back') navigation.goBack();
  };

  const onCommand = (command: Command) => {
    setLastCommand(command);
    if (command === 'on' || command === 'off') { setFocus(command); run(command); }
    if (command === 'left') setFocus('on');
    if (command === 'right') setFocus('off');
    if (command === 'down') setFocus('back');
    if (command === 'up' && focus === 'back') setFocus('on');
    if (command === 'yes') run(focus);
    if (command === 'no') navigation.goBack();
  };

  return (
    <AppShell eyebrow="DEVICE CONTROL" title={device.name}>
      <View style={styles.deviceHero}>
        <Text style={styles.deviceGlyph}>◉</Text>
        <View style={styles.deviceText}><Text style={styles.deviceName}>Current state</Text><Text style={styles.deviceMeta}>{device.detail}</Text></View>
        <View style={[styles.powerPill, device.on ? styles.powerOn : styles.powerOff]}>
          <Text style={[styles.powerText, !device.on && styles.badgeTextOff]}>{device.on ? 'ON' : 'OFF'}</Text>
        </View>
      </View>
      <View style={styles.actionRow}>
        <ActionCard title="Turn On" detail="Say ON" focused={focus === 'on'} onPress={() => { setFocus('on'); run('on'); }} />
        <ActionCard title="Turn Off" detail="Say OFF" focused={focus === 'off'} onPress={() => { setFocus('off'); run('off'); }} />
      </View>
      <Pressable style={[styles.routineCard, focus === 'back' && styles.focused]} onPress={() => run('back')}>
        <View><Text style={styles.cardTitle}>Back to dashboard</Text><Text style={styles.cardDetail}>Say NO, or YES on this option</Text></View>
        <Text style={styles.cardArrow}>←</Text>
      </Pressable>
      <VoiceControl onCommand={onCommand} />
      <View style={styles.quickRow}>
        {(['on', 'off', 'yes', 'no'] as Command[]).map((c) => <QuickCommand key={c} command={c} onPress={() => onCommand(c)} />)}
      </View>
      <Text style={styles.hint}>
        ON/OFF control the device; YES runs the highlighted action; NO returns.{lastCommand ? `  Last: ${commandLabels[lastCommand]}` : ''}
      </Text>
    </AppShell>
  );
}

function ActionCard({ title, detail, focused, onPress }: { title: string; detail: string; focused: boolean; onPress: () => void }) {
  return (
    <Pressable style={[styles.actionCard, focused && styles.focused]} onPress={onPress}>
      <Text style={styles.deviceName}>{title}</Text>
      <Text style={styles.cardDetail}>{detail}</Text>
    </Pressable>
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

function RoutineCard({ name, detail, onPress }: { name: string; detail: string; onPress: () => void }) {
  return <Pressable style={styles.routineCard} onPress={onPress}><View><Text style={styles.cardTitle}>{name}</Text><Text style={styles.cardDetail}>{detail}</Text></View><Text style={styles.cardArrow}>→</Text></Pressable>;
}

// Con onPress, el botón simula el comando: sirve para probar la navegación sin el modelo
function QuickCommand({ command, onPress }: { command: Command; onPress?: () => void }) {
  return <Pressable style={styles.quickCommand} onPress={onPress} disabled={!onPress}><Text style={styles.quickText}>{commandLabels[command]}</Text></Pressable>;
}

export default function App() {
  return <SafeAreaProvider><DevicesProvider><NavigationContainer><Stack.Navigator screenOptions={{ headerShown: false }}><Stack.Screen name="Home" component={HomeScreen} /><Stack.Screen name="Device" component={DeviceScreen} /><Stack.Screen name="Routines" component={RoutinesScreen} /><Stack.Screen name="Confirm" component={ConfirmScreen} /></Stack.Navigator></NavigationContainer></DevicesProvider></SafeAreaProvider>;
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
  micButtonDisabled: { opacity: 0.4 },
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
  focused: { borderWidth: 2, borderColor: '#d7ff64' },
  homeGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', rowGap: 12 },
  homeCard: { width: '48.5%', minHeight: 132, backgroundColor: '#1a201e', padding: 16, borderRadius: 6, borderWidth: 2, borderColor: 'transparent' },
  badge: { alignSelf: 'flex-start', marginTop: 'auto', paddingHorizontal: 10, paddingVertical: 5, borderRadius: 12 },
  badgeOpen: { backgroundColor: '#65d9ff' },
  badgeTextOff: { color: '#c8cec8' },
  deviceText: { flexShrink: 1 },
  actionRow: { flexDirection: 'row', gap: 12, marginTop: 14 },
  actionCard: { flex: 1, minHeight: 96, backgroundColor: '#1a201e', padding: 16, borderRadius: 6, borderWidth: 2, borderColor: 'transparent', justifyContent: 'center' },
});
