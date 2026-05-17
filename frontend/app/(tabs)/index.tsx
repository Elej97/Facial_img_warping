import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { Animated, Pressable, StyleSheet, View, useWindowDimensions } from 'react-native';

import { STUDIO, StudioCard, StudioScreen } from '@/components/studio-shell';
import { ThemedText } from '@/components/themed-text';
import { useAuth } from '@/hooks/use-auth';
import { useColorScheme } from '@/hooks/use-color-scheme';

function DemoCard({ compact = false }: { compact?: boolean }) {
  const [processed, setProcessed] = useState(false);
  const fade = useRef(new Animated.Value(1)).current;
  const colorScheme = useColorScheme() ?? 'dark';
  const isDark = colorScheme === 'dark';

  useEffect(() => {
    const interval = setInterval(() => {
      Animated.sequence([
        Animated.timing(fade, { toValue: 0, duration: 360, useNativeDriver: true }),
        Animated.timing(fade, { toValue: 1, duration: 520, useNativeDriver: true }),
      ]).start();
      setProcessed((current) => !current);
    }, 2600);
    return () => clearInterval(interval);
  }, [fade]);

  return (
    <StudioCard style={[styles.demoCard, !isDark ? styles.demoCardLight : null, compact ? styles.demoCardCompact : null]}>
      <View style={styles.demoBadge}>
        <ThemedText style={[styles.demoBadgeText, !isDark ? styles.darkText : null]}>DEMO SHOWCASE</ThemedText>
      </View>
      <Animated.View style={{ opacity: fade }}>
        <ThemedText style={[styles.demoCenterText, !isDark ? styles.demoCenterTextLight : null]}>
          {processed ? 'DÖNÜŞTÜRÜLDÜ' : 'ORIJINAL'}
        </ThemedText>
      </Animated.View>
      <View style={styles.demoFooter}>
        <View>
          <ThemedText style={[styles.demoTitle, !isDark ? styles.darkText : null]}>Portrait Transformation</ThemedText>
          <ThemedText style={styles.demoAccent}>AI MORPHED</ThemedText>
        </View>
        <Ionicons name="flash-outline" size={16} color="#A855F7" />
      </View>
    </StudioCard>
  );
}

export default function HomeScreen() {
  const router = useRouter();
  const { width } = useWindowDimensions();
  const { userName } = useAuth();
  const colorScheme = useColorScheme() ?? 'dark';
  const isDark = colorScheme === 'dark';
  const isWide = width >= 980;
  const firstName = userName?.trim().split(/\s+/)[0];

  return (
    <StudioScreen>
      <View style={[styles.page, isWide ? styles.pageWide : null]}>
        <View style={styles.heroCopy}>
          <View style={styles.welcomeSection}>
            <View style={styles.welcomeContent}>
              <ThemedText style={[styles.welcomeText, !isDark ? styles.welcomeTextLight : null]}>
                Hoşgeldiniz{firstName ? ', ' + firstName : ''}
              </ThemedText>
            </View>
            {!userName && (
              <Pressable
                style={[styles.loginButton, !isDark ? styles.loginButtonLight : null]}
                onPress={() => router.push('/auth')}
              >
                <ThemedText style={styles.loginButtonText}>Giriş Yap</ThemedText>
              </Pressable>
            )}
          </View>

          <View style={styles.kicker}>
            <Ionicons name="flash" size={13} color="#C084FC" />
            <ThemedText style={styles.kickerText}>PROFESYONEL DÖNÜŞÜM</ThemedText>
          </View>

          <ThemedText style={[styles.heroTitle, !isDark ? styles.heroTitleLight : null]}>
            Kendi tarzınızı{'\n'}
            <ThemedText style={styles.gradientWord}>yansıtan{'\n'}içerikler</ThemedText>
            {'\n'}oluşturun.
          </ThemedText>

          <ThemedText style={[styles.heroDesc, !isDark ? styles.heroDescLight : null]}>
            Yüz ifadelerini değiştir, yaşını ayarla ve görüntülerini hayal ettiğin şekilde dönüştür.
            Profesyonel araçlar cebinizde.
          </ThemedText>

          <View style={styles.heroActions}>
            <Pressable style={styles.primaryButton} onPress={() => router.push('/create')}>
              <ThemedText style={styles.primaryButtonText}>Hemen Başla</ThemedText>
            </Pressable>
            <Pressable style={[styles.secondaryButton, !isDark ? styles.secondaryButtonLight : null]} onPress={() => router.push('/create')}>
              <ThemedText style={[styles.secondaryButtonText, !isDark ? styles.darkText : null]}>Özellikleri Keşfet</ThemedText>
            </Pressable>
          </View>
        </View>

        {isWide ? (
          <View style={styles.demoGrid}>
            <DemoCard />
            <DemoCard compact />
          </View>
        ) : null}
      </View>
    </StudioScreen>
  );
}

const styles = StyleSheet.create({
  page: {
    flex: 1,
    paddingHorizontal: 36,
    paddingVertical: 44,
    justifyContent: 'center',
    gap: 42,
  },
  pageWide: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 74,
  },
  heroCopy: {
    flex: 1,
    maxWidth: 560,
    gap: 28,
  },
  welcomeSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    flexWrap: 'wrap',
    marginBottom: 8,
  },
  welcomeContent: {
    flexShrink: 1,
  },
  welcomeText: {
    color: STUDIO.text,
    fontSize: 42,
    lineHeight: 48,
    fontWeight: '900',
    letterSpacing: 0,
  },
  welcomeTextLight: {
    color: STUDIO.lightText,
  },
  loginButton: {
    minHeight: 60,
    paddingHorizontal: 28,
    paddingVertical: 14,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(168,85,247,0.3)',
    borderWidth: 1,
    borderColor: 'rgba(168,85,247,0.5)',
  },
  loginButtonLight: {
    backgroundColor: 'rgba(168,85,247,0.15)',
  },
  loginButtonText: {
    color: '#C084FC',
    fontSize: 20,
    fontWeight: '900',
  },
  kicker: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 6,
    backgroundColor: 'rgba(168,85,247,0.24)',
    borderWidth: 1,
    borderColor: 'rgba(168,85,247,0.34)',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  kickerText: {
    color: '#C084FC',
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1.4,
  },
  heroTitle: {
    color: STUDIO.text,
    fontSize: 74,
    lineHeight: 72,
    fontWeight: '900',
    letterSpacing: 0,
  },
  heroTitleLight: {
    color: STUDIO.lightText,
  },
  gradientWord: {
    color: '#DB66E8',
    fontSize: 74,
    lineHeight: 72,
    fontWeight: '900',
  },
  heroDesc: {
    color: '#98A7C5',
    fontSize: 21,
    lineHeight: 28,
    maxWidth: 490,
  },
  heroDescLight: {
    color: STUDIO.lightMuted,
  },
  heroActions: {
    flexDirection: 'row',
    gap: 16,
    flexWrap: 'wrap',
    marginTop: 22,
  },
  primaryButton: {
    minWidth: 170,
    height: 60,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F0F0F2',
  },
  primaryButtonText: {
    color: '#000000',
    fontWeight: '900',
  },
  secondaryButton: {
    minWidth: 178,
    height: 60,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.10)',
  },
  secondaryButtonLight: {
    backgroundColor: 'rgba(255,255,255,0.72)',
    borderWidth: 1,
    borderColor: STUDIO.lightBorder,
  },
  secondaryButtonText: {
    color: '#FFFFFF',
    fontWeight: '900',
  },
  demoGrid: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 24,
  },
  demoCard: {
    width: 265,
    height: 352,
    borderRadius: 38,
    padding: 24,
    justifyContent: 'space-between',
    backgroundColor: 'rgba(31,31,34,0.88)',
  },
  demoCardLight: {
    backgroundColor: 'rgba(255,255,255,0.82)',
  },
  demoCardCompact: {
    marginTop: 2,
  },
  demoBadge: {
    alignSelf: 'flex-end',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.12)',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  demoBadgeText: {
    color: '#FFFFFF',
    fontSize: 8,
    fontWeight: '900',
  },
  darkText: {
    color: '#111217',
  },
  demoCenterText: {
    color: 'rgba(255,255,255,0.14)',
    textAlign: 'center',
    fontWeight: '900',
    letterSpacing: 1,
  },
  demoCenterTextLight: {
    color: 'rgba(17,18,23,0.16)',
  },
  demoFooter: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
  },
  demoTitle: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '900',
  },
  demoAccent: {
    color: '#C084FC',
    fontSize: 10,
    fontWeight: '900',
    marginTop: 8,
  },
});
