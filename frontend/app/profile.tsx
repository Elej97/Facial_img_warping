import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, View } from 'react-native';

import { STUDIO, StudioCard, StudioScreen } from '@/components/studio-shell';
import { ThemedText } from '@/components/themed-text';
import { useAuth } from '@/hooks/use-auth';
import { useColorScheme } from '@/hooks/use-color-scheme';

const API_BASE = (process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:3000').replace(/\/$/, '');

type Upload = {
  id: number;
  created_at: string;
};

type StatCardProps = {
  label: string;
  value: string;
  icon: keyof typeof Ionicons.glyphMap;
};

type QuickActionProps = {
  title: string;
  subtitle: string;
  icon: keyof typeof Ionicons.glyphMap;
  onPress: () => void;
};

function StatCard({ label, value, icon }: StatCardProps) {
  const colorScheme = useColorScheme() ?? 'dark';
  const isDark = colorScheme === 'dark';

  return (
    <StudioCard style={styles.statCard}>
      <View style={[styles.statIcon, !isDark ? styles.softIconLight : null]}>
        <Ionicons name={icon} size={20} color="#C084FC" />
      </View>
      <ThemedText style={[styles.statValue, !isDark ? styles.darkText : null]}>{value}</ThemedText>
      <ThemedText style={[styles.statLabel, !isDark ? styles.mutedTextLight : null]}>{label}</ThemedText>
    </StudioCard>
  );
}

function QuickAction({ title, subtitle, icon, onPress }: QuickActionProps) {
  const colorScheme = useColorScheme() ?? 'dark';
  const isDark = colorScheme === 'dark';

  return (
    <Pressable onPress={onPress} style={[styles.actionRow, !isDark ? styles.actionRowLight : null]}>
      <View style={[styles.actionIcon, !isDark ? styles.softIconLight : null]}>
        <Ionicons name={icon} size={20} color="#C084FC" />
      </View>
      <View style={styles.actionCopy}>
        <ThemedText style={[styles.actionTitle, !isDark ? styles.darkText : null]}>{title}</ThemedText>
        <ThemedText style={[styles.actionSubtitle, !isDark ? styles.mutedTextLight : null]}>{subtitle}</ThemedText>
      </View>
      <Ionicons name="chevron-forward" size={18} color={isDark ? '#7D88A0' : STUDIO.lightMuted} />
    </Pressable>
  );
}

export default function ProfileScreen() {
  const router = useRouter();
  const { userName, token, signOut } = useAuth();
  const colorScheme = useColorScheme() ?? 'dark';
  const isDark = colorScheme === 'dark';
  const [uploads, setUploads] = useState<Upload[]>([]);

  const initials = useMemo(() => {
    const cleanName = userName?.trim() || 'U';
    return cleanName
      .split(/\s+/)
      .slice(0, 2)
      .map((part) => part[0]?.toLocaleUpperCase('tr-TR'))
      .join('');
  }, [userName]);

  const latestDate = useMemo(() => {
    const latest = uploads[0]?.created_at;
    return latest ? new Date(latest).toLocaleDateString('tr-TR') : 'Henüz yok';
  }, [uploads]);

  const loadProfileStats = useCallback(async () => {
    if (!token) {
      setUploads([]);
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/api/auth/library`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await response.json().catch(() => ({}));
      setUploads(response.ok && Array.isArray(data.uploads) ? data.uploads : []);
    } catch (err) {
      console.warn('Profile stats unavailable:', err);
      setUploads([]);
    }
  }, [token]);

  useEffect(() => {
    if (!userName) {
      router.replace('/auth');
    }
  }, [router, userName]);

  useEffect(() => {
    loadProfileStats();
  }, [loadProfileStats]);

  if (!userName) {
    return null;
  }

  return (
    <StudioScreen>
      <ScrollView contentContainerStyle={styles.page} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <View>
            <ThemedText style={[styles.kicker, !isDark ? styles.kickerLight : null]}>HESAP MERKEZİ</ThemedText>
            <ThemedText style={[styles.title, !isDark ? styles.titleLight : null]}>Profil</ThemedText>
            <ThemedText style={[styles.subtitle, !isDark ? styles.subtitleLight : null]}>
              Hesabınızı, üretim geçmişinizi ve çalışma alanı tercihlerinizi buradan yönetin.
            </ThemedText>
          </View>
        </View>

        <View style={styles.layout}>
          <StudioCard style={styles.heroCard}>
            <View style={styles.profileTop}>
              <View style={styles.avatar}>
                <ThemedText style={styles.avatarText}>{initials}</ThemedText>
              </View>
              <View style={styles.profileCopy}>
                <ThemedText style={[styles.userName, !isDark ? styles.darkText : null]}>{userName}</ThemedText>
                <ThemedText style={[styles.userStatus, !isDark ? styles.mutedTextLight : null]}>Aktif stüdyo hesabı</ThemedText>
              </View>
            </View>

            <View style={[styles.memberBadge, !isDark ? styles.memberBadgeLight : null]}>
              <Ionicons name="sparkles-outline" size={16} color="#C084FC" />
              <ThemedText style={styles.memberBadgeText}>PROFESYONEL DÖNÜŞÜM</ThemedText>
            </View>

            <View style={styles.infoList}>
              <View style={styles.infoRow}>
                <ThemedText style={[styles.infoLabel, !isDark ? styles.mutedTextLight : null]}>Plan</ThemedText>
                <ThemedText style={[styles.infoValue, !isDark ? styles.darkText : null]}>Studio Free</ThemedText>
              </View>
              <View style={styles.infoRow}>
                <ThemedText style={[styles.infoLabel, !isDark ? styles.mutedTextLight : null]}>Son içerik</ThemedText>
                <ThemedText style={[styles.infoValue, !isDark ? styles.darkText : null]}>{latestDate}</ThemedText>
              </View>
              <View style={styles.infoRow}>
                <ThemedText style={[styles.infoLabel, !isDark ? styles.mutedTextLight : null]}>Oturum</ThemedText>
                <ThemedText style={[styles.infoValue, !isDark ? styles.darkText : null]}>Güvenli</ThemedText>
              </View>
            </View>

            <Pressable
              onPress={() => {
                signOut();
                router.replace('/');
              }}
              style={[styles.signOutButton, !isDark ? styles.signOutButtonLight : null]}>
              <Ionicons name="log-out-outline" size={18} color="#F87171" />
              <ThemedText style={styles.signOutText}>Oturumu Kapat</ThemedText>
            </Pressable>
          </StudioCard>

          <View style={styles.contentColumn}>
            <View style={styles.statsGrid}>
              <StatCard label="Kaydedilen içerik" value={String(uploads.length)} icon="images-outline" />
              <StatCard label="Favori efekt" value="Morfing" icon="flash-outline" />
              <StatCard label="Çalışma modu" value="Web" icon="desktop-outline" />
            </View>

            <StudioCard style={styles.panel}>
              <View style={styles.panelHeader}>
                <ThemedText style={[styles.panelTitle, !isDark ? styles.darkText : null]}>Hızlı İşlemler</ThemedText>
                <ThemedText style={[styles.panelHint, !isDark ? styles.mutedTextLight : null]}>Profilinden sık kullanılan alanlara geç.</ThemedText>
              </View>

              <View style={styles.actions}>
                <QuickAction
                  icon="add-circle-outline"
                  title="Yeni içerik oluştur"
                  subtitle="Fotoğraf seçip dönüşüm akışına geç"
                  onPress={() => router.push('/create')}
                />
                <QuickAction
                  icon="library-outline"
                  title="Kütüphaneyi aç"
                  subtitle="Kaydettiğin içerikleri görüntüle"
                  onPress={() => router.push('/library')}
                />
                <QuickAction
                  icon="settings-outline"
                  title="Stüdyo ayarları"
                  subtitle="Kalite, format ve görünüm tercihlerini düzenle"
                  onPress={() => router.push('/settings')}
                />
              </View>
            </StudioCard>
          </View>
        </View>
      </ScrollView>
    </StudioScreen>
  );
}

const styles = StyleSheet.create({
  page: {
    paddingHorizontal: 56,
    paddingTop: 44,
    paddingBottom: 72,
    gap: 34,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 24,
  },
  kicker: {
    color: '#C084FC',
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1.2,
    marginBottom: 8,
  },
  kickerLight: {
    color: STUDIO.accent,
  },
  title: {
    color: STUDIO.text,
    fontSize: 54,
    lineHeight: 58,
    fontWeight: '900',
  },
  titleLight: {
    color: STUDIO.lightText,
  },
  subtitle: {
    color: '#7584A3',
    fontSize: 16,
    lineHeight: 23,
    fontWeight: '700',
    maxWidth: 640,
  },
  subtitleLight: {
    color: STUDIO.lightMuted,
  },
  layout: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 24,
    flexWrap: 'wrap',
  },
  heroCard: {
    width: 360,
    minHeight: 520,
    padding: 26,
    borderRadius: 30,
    justifyContent: 'space-between',
    gap: 24,
  },
  profileTop: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  avatar: {
    width: 86,
    height: 86,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: STUDIO.accent,
    shadowColor: STUDIO.accent,
    shadowOpacity: 0.42,
    shadowRadius: 22,
    shadowOffset: { width: 0, height: 16 },
  },
  avatarText: {
    color: '#FFFFFF',
    fontSize: 30,
    fontWeight: '900',
  },
  profileCopy: {
    flex: 1,
    gap: 5,
  },
  userName: {
    color: '#FFFFFF',
    fontSize: 25,
    lineHeight: 30,
    fontWeight: '900',
  },
  userStatus: {
    color: '#8EA0C0',
    fontSize: 13,
    fontWeight: '800',
  },
  memberBadge: {
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 999,
    backgroundColor: 'rgba(168,85,247,0.16)',
    borderWidth: 1,
    borderColor: 'rgba(168,85,247,0.28)',
  },
  memberBadgeLight: {
    backgroundColor: 'rgba(168,85,247,0.10)',
  },
  memberBadgeText: {
    color: '#C084FC',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.7,
  },
  infoList: {
    gap: 14,
  },
  infoRow: {
    minHeight: 48,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.08)',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 16,
  },
  infoLabel: {
    color: '#7D88A0',
    fontSize: 13,
    fontWeight: '800',
  },
  infoValue: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '900',
  },
  signOutButton: {
    height: 52,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
    backgroundColor: 'rgba(248,113,113,0.10)',
    borderWidth: 1,
    borderColor: 'rgba(248,113,113,0.18)',
  },
  signOutButtonLight: {
    backgroundColor: 'rgba(248,113,113,0.08)',
  },
  signOutText: {
    color: '#F87171',
    fontSize: 14,
    fontWeight: '900',
  },
  contentColumn: {
    flex: 1,
    minWidth: 360,
    gap: 24,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
  },
  statCard: {
    flexGrow: 1,
    flexBasis: 180,
    minHeight: 150,
    padding: 20,
    borderRadius: 26,
    justifyContent: 'space-between',
  },
  statIcon: {
    width: 42,
    height: 42,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(168,85,247,0.16)',
  },
  statValue: {
    color: '#FFFFFF',
    fontSize: 28,
    fontWeight: '900',
  },
  statLabel: {
    color: '#8EA0C0',
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 0.4,
  },
  panel: {
    padding: 24,
    borderRadius: 30,
    gap: 22,
  },
  panelHeader: {
    gap: 6,
  },
  panelTitle: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '900',
  },
  panelHint: {
    color: '#8EA0C0',
    fontSize: 14,
    fontWeight: '700',
  },
  actions: {
    gap: 12,
  },
  actionRow: {
    minHeight: 76,
    borderRadius: 22,
    paddingHorizontal: 16,
    paddingVertical: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
  },
  actionRowLight: {
    backgroundColor: 'rgba(255,255,255,0.72)',
    borderColor: STUDIO.lightBorder,
  },
  actionIcon: {
    width: 42,
    height: 42,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(168,85,247,0.16)',
  },
  actionCopy: {
    flex: 1,
    gap: 4,
  },
  actionTitle: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
  actionSubtitle: {
    color: '#8EA0C0',
    fontSize: 12,
    lineHeight: 17,
    fontWeight: '700',
  },
  softIconLight: {
    backgroundColor: 'rgba(168,85,247,0.10)',
  },
  darkText: {
    color: STUDIO.lightText,
  },
  mutedTextLight: {
    color: STUDIO.lightMuted,
  },
});
