import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, View } from 'react-native';

import { STUDIO, StudioCard, StudioScreen } from '@/components/studio-shell';
import { ThemedText } from '@/components/themed-text';
import { useAuth } from '@/hooks/use-auth';
import { useColorScheme } from '@/hooks/use-color-scheme';

const API_BASE = (process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:3000').replace(/\/$/, '');

type Upload = {
  id: number;
  file_name: string;
  created_at: string;
};

export default function LibraryScreen() {
  const router = useRouter();
  const { userName, token, isLoading: authLoading } = useAuth();
  const colorScheme = useColorScheme() ?? 'dark';
  const isDark = colorScheme === 'dark';
  
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const loadUploads = useCallback(async () => {
    if (!token) {
      setUploads([]);
      setError('');
      return;
    }

    setIsLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/api/auth/library`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      const data = await response.json().catch(() => ({}));

      if (response.ok) {
        setUploads(Array.isArray(data.uploads) ? data.uploads : []);
        return;
      }

      console.warn('Library request failed:', data.error || response.status);
      setUploads([]);
    } catch (err) {
      console.error('Load uploads error:', err);
      setUploads([]);
    } finally {
      setIsLoading(false);
    }
  }, [token]);

  useEffect(() => {
    loadUploads();
  }, [loadUploads]);

  if (authLoading) {
    return (
      <StudioScreen>
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color="#C084FC" />
        </View>
      </StudioScreen>
    );
  }

  if (!userName) {
    return (
      <StudioScreen>
        <View style={styles.centerContainer}>
          <View style={[styles.loginPrompt, !isDark ? styles.loginPromptLight : null]}>
            <Ionicons name="lock-closed" size={48} color="#C084FC" />
            <ThemedText style={[styles.loginPromptTitle, !isDark ? styles.loginPromptTitleLight : null]}>
              Kütüphanenizi Açın
            </ThemedText>
            <ThemedText style={[styles.loginPromptText, !isDark ? styles.loginPromptTextLight : null]}>
              Daha önce oluşturduğunuz içerikleri görmek için lütfen giriş yapın.
            </ThemedText>
            <Pressable
              style={styles.loginPromptButton}
              onPress={() => router.push('/auth')}
            >
              <ThemedText style={styles.loginPromptButtonText}>Giriş Yap</ThemedText>
            </Pressable>
          </View>
        </View>
      </StudioScreen>
    );
  }

  return (
    <StudioScreen>
      <ScrollView contentContainerStyle={styles.page} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <View>
            <ThemedText style={[styles.title, !isDark ? styles.titleLight : null]}>Kütüphane</ThemedText>
            <ThemedText style={[styles.subtitle, !isDark ? styles.subtitleLight : null]}>
              {uploads.length === 0 ? 'Henüz içerik yok' : `${uploads.length} içerik kaydedildi`}
            </ThemedText>
          </View>
          <View style={styles.filters}>
            {['TÜMÜ', 'MORFING', 'FİLTRELER', 'FAVORİLER'].map((filter, index) => (
              <Pressable key={filter} style={[styles.filterPill, !isDark ? styles.filterPillLight : null, index === 0 ? styles.filterPillActive : null]}>
                <ThemedText style={[styles.filterText, index === 0 ? styles.filterTextActive : null]}>{filter}</ThemedText>
              </Pressable>
            ))}
          </View>
        </View>

        {isLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#C084FC" />
          </View>
        ) : error ? (
          <View style={styles.errorContainer}>
            <ThemedText style={[styles.errorText, !isDark ? styles.errorTextLight : null]}>
              {error}
            </ThemedText>
          </View>
        ) : uploads.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Ionicons name="image-outline" size={48} color="rgba(255,255,255,0.18)" />
            <ThemedText style={styles.emptyText}>Henüz içerik oluşturmadınız</ThemedText>
            <Pressable
              style={styles.createButton}
              onPress={() => router.push('/create')}
            >
              <ThemedText style={styles.createButtonText}>Yeni Oluştur</ThemedText>
            </Pressable>
          </View>
        ) : (
          <View style={styles.grid}>
            {uploads.map((item) => (
              <StudioCard key={item.id} style={styles.itemCard}>
                <View style={[styles.thumb, !isDark ? styles.thumbLight : null]}>
                  <Ionicons name="image-outline" size={46} color="rgba(255,255,255,0.18)" />
                  <View style={styles.thumbActions}>
                    <Pressable style={styles.inspectButton}>
                      <ThemedText style={styles.inspectText}>İncele</ThemedText>
                    </Pressable>
                    <Pressable style={styles.downloadButton}>
                      <Ionicons name="download-outline" size={14} color="#FFFFFF" />
                    </Pressable>
                  </View>
                </View>
                <View style={styles.itemMeta}>
                  <ThemedText style={[styles.itemTitle, !isDark ? styles.itemTitleLight : null]}>
                    {item.file_name}
                  </ThemedText>
                  <View style={styles.itemFooter}>
                    <ThemedText style={styles.itemType}>MORFING</ThemedText>
                    <ThemedText style={styles.itemDate}>
                      {new Date(item.created_at).toLocaleDateString('tr-TR')}
                    </ThemedText>
                  </View>
                </View>
              </StudioCard>
            ))}

            <Pressable style={[styles.newCard, !isDark ? styles.newCardLight : null]} onPress={() => router.push('/create')}>
              <View style={styles.newIcon}>
                <Ionicons name="add-circle-outline" size={26} color="#C084FC" />
              </View>
              <ThemedText style={styles.newText}>YENİ OLUŞTUR</ThemedText>
            </Pressable>
          </View>
        )}
      </ScrollView>
    </StudioScreen>
  );
}

const styles = StyleSheet.create({
  page: {
    paddingHorizontal: 56,
    paddingTop: 42,
    paddingBottom: 64,
    gap: 38,
  },
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
  },
  loginPrompt: {
    alignItems: 'center',
    gap: 16,
    padding: 32,
    borderRadius: 24,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  loginPromptLight: {
    backgroundColor: 'rgba(15,23,42,0.05)',
    borderColor: 'rgba(15,23,42,0.1)',
  },
  loginPromptTitle: {
    fontSize: 20,
    fontWeight: '900',
    color: '#FFFFFF',
  },
  loginPromptTitleLight: {
    color: STUDIO.lightText,
  },
  loginPromptText: {
    fontSize: 16,
    textAlign: 'center',
    color: '#A0AEC0',
  },
  loginPromptTextLight: {
    color: STUDIO.lightMuted,
  },
  loginPromptButton: {
    marginTop: 12,
    paddingHorizontal: 24,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: '#A855F7',
  },
  loginPromptButtonText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 14,
  },
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorContainer: {
    alignItems: 'center',
    padding: 20,
    borderRadius: 12,
    backgroundColor: 'rgba(239,68,68,0.1)',
  },
  errorText: {
    color: '#F87171',
    fontSize: 14,
    textAlign: 'center',
  },
  errorTextLight: {
    color: '#DC2626',
  },
  emptyContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  emptyText: {
    fontSize: 16,
    color: '#A0AEC0',
  },
  createButton: {
    marginTop: 12,
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#A855F7',
  },
  createButtonText: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 24,
    flexWrap: 'wrap',
  },
  title: {
    color: STUDIO.text,
    fontSize: 50,
    lineHeight: 56,
    fontWeight: '900',
  },
  titleLight: {
    color: STUDIO.lightText,
  },
  subtitle: {
    color: '#7584A3',
    fontSize: 16,
    fontWeight: '700',
  },
  subtitleLight: {
    color: STUDIO.lightMuted,
  },
  filters: {
    flexDirection: 'row',
    gap: 10,
    flexWrap: 'wrap',
  },
  filterPill: {
    height: 36,
    minWidth: 96,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  filterPillLight: {
    backgroundColor: 'rgba(255,255,255,0.72)',
  },
  filterPillActive: {
    backgroundColor: STUDIO.accent,
  },
  filterText: {
    color: '#8EA0C0',
    fontSize: 12,
    fontWeight: '900',
  },
  filterTextActive: {
    color: '#FFFFFF',
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 24,
  },
  itemCard: {
    width: 278,
    height: 333,
    padding: 16,
    borderRadius: 38,
    justifyContent: 'space-between',
  },
  thumb: {
    flex: 1,
    borderRadius: 28,
    backgroundColor: 'rgba(0,0,0,0.28)',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    overflow: 'hidden',
  },
  thumbLight: {
    backgroundColor: 'rgba(15,23,42,0.05)',
  },
  thumbActions: {
    position: 'absolute',
    left: 16,
    right: 16,
    bottom: 18,
    flexDirection: 'row',
    gap: 8,
  },
  inspectButton: {
    flex: 1,
    height: 34,
    borderRadius: 11,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.10)',
  },
  inspectText: {
    color: 'rgba(255,255,255,0.55)',
    fontSize: 11,
    fontWeight: '900',
  },
  downloadButton: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.10)',
  },
  itemMeta: {
    paddingHorizontal: 8,
    paddingTop: 16,
    gap: 8,
  },
  itemTitle: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
  itemTitleLight: {
    color: STUDIO.lightText,
  },
  itemFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  itemType: {
    color: '#B225FF',
    fontSize: 10,
    fontWeight: '900',
  },
  itemDate: {
    color: '#7180A0',
    fontSize: 10,
    fontWeight: '700',
  },
  newCard: {
    width: 278,
    height: 84,
    borderRadius: 38,
    borderWidth: 2,
    borderStyle: 'dashed',
    borderColor: 'rgba(255,255,255,0.13)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  newCardLight: {
    borderColor: 'rgba(15,23,42,0.16)',
  },
  newIcon: {
    width: 48,
    height: 48,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(168,85,247,0.16)',
  },
  newText: {
    color: '#7180A0',
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 1,
  },
});
