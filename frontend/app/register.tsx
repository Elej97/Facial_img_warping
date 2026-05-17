import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, TextInput, View } from 'react-native';

import { StudioScreen } from '@/components/studio-shell';
import { ThemedText } from '@/components/themed-text';
import { useAuth } from '@/hooks/use-auth';
import { useColorScheme } from '@/hooks/use-color-scheme';

export default function RegisterScreen() {
  const router = useRouter();
  const { signUp } = useAuth();
  const colorScheme = useColorScheme() ?? 'dark';
  const isDark = colorScheme === 'dark';
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordRepeat, setPasswordRepeat] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const onSubmit = async () => {
    setError('');

    if (!fullName.trim() || !email.trim() || !password.trim() || !passwordRepeat.trim()) {
      setError('Tüm alanlar gereklidir');
      return;
    }

    if (password !== passwordRepeat) {
      setError('Parolalar eşleşmiyor');
      return;
    }

    if (password.length < 6) {
      setError('Parola en az 6 karakter olmalıdır');
      return;
    }

    setIsLoading(true);
    const success = await signUp(email.trim(), fullName.trim(), password.trim());
    setIsLoading(false);

    if (success) {
      router.replace('/');
    } else {
      setError('Kayıt yapılamadı. Email zaten kullanımda olabilir.');
    }
  };

  return (
    <StudioScreen withNav={false}>
      <View style={[styles.backdrop, !isDark ? styles.backdropLight : null]}>
        <View style={styles.blurBlockOne} />
        <View style={styles.blurBlockTwo} />
        <View style={[styles.modalCard, isDark ? styles.modalCardDark : null]}>
          <Pressable style={styles.closeButton} onPress={() => router.back()}>
            <Ionicons name="close" size={24} color="#737B8D" />
          </Pressable>

          <ThemedText style={[styles.title, isDark ? styles.titleDark : null]}>Hesap Oluştur</ThemedText>
          <ThemedText style={[styles.subtitle, isDark ? styles.subtitleDark : null]}>FaceMorph topluluğuna katılarak çalışmalarını kaydet.</ThemedText>

          <View style={styles.field}>
            <ThemedText style={[styles.label, isDark ? styles.labelDark : null]}>İSİM SOYISIM</ThemedText>
            <TextInput
              value={fullName}
              onChangeText={setFullName}
              placeholder="Ahmet Yılmaz"
              placeholderTextColor={isDark ? '#737B8D' : '#8A8A8A'}
              editable={!isLoading}
              style={[styles.input, isDark ? styles.inputDark : null]}
            />
          </View>

          <View style={styles.field}>
            <ThemedText style={[styles.label, isDark ? styles.labelDark : null]}>E-POSTA</ThemedText>
            <TextInput
              value={email}
              onChangeText={setEmail}
              placeholder="ahmet@example.com"
              placeholderTextColor={isDark ? '#737B8D' : '#8A8A8A'}
              autoCapitalize="none"
              editable={!isLoading}
              style={[styles.input, isDark ? styles.inputDark : null]}
            />
          </View>

          <View style={styles.field}>
            <ThemedText style={[styles.label, isDark ? styles.labelDark : null]}>PAROLA</ThemedText>
            <TextInput
              value={password}
              onChangeText={setPassword}
              placeholder="••••••••"
              placeholderTextColor={isDark ? '#737B8D' : '#8A8A8A'}
              secureTextEntry
              editable={!isLoading}
              style={[styles.input, isDark ? styles.inputDark : null]}
            />
          </View>

          <View style={styles.field}>
            <ThemedText style={[styles.label, isDark ? styles.labelDark : null]}>PAROLA TEKRAR</ThemedText>
            <TextInput
              value={passwordRepeat}
              onChangeText={setPasswordRepeat}
              placeholder="••••••••"
              placeholderTextColor={isDark ? '#737B8D' : '#8A8A8A'}
              secureTextEntry
              editable={!isLoading}
              style={[styles.input, isDark ? styles.inputDark : null]}
            />
          </View>

          {error ? (
            <ThemedText style={[styles.errorText, !isDark ? styles.errorTextLight : null]}>
              {error}
            </ThemedText>
          ) : null}

          <Pressable
            style={[styles.submitButton, isLoading ? styles.submitButtonDisabled : null]}
            onPress={onSubmit}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color="#FFFFFF" />
            ) : (
              <ThemedText style={styles.submitText}>Kayıt Ol</ThemedText>
            )}
          </Pressable>

          <Pressable style={styles.registerLink} onPress={() => router.push('/auth')} disabled={isLoading}>
            <ThemedText style={styles.registerText}>Zaten hesabın var mı? Giriş Yap</ThemedText>
          </Pressable>
        </View>
      </View>
    </StudioScreen>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(230,230,235,0.42)',
  },
  backdropLight: {
    backgroundColor: 'rgba(235,235,242,0.62)',
  },
  blurBlockOne: {
    position: 'absolute',
    left: 300,
    top: 86,
    width: 230,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.28)',
  },
  blurBlockTwo: {
    position: 'absolute',
    bottom: 92,
    right: 250,
    width: 180,
    height: 42,
    borderRadius: 22,
    backgroundColor: 'rgba(160,32,240,0.48)',
  },
  modalCard: {
    width: 448,
    borderRadius: 48,
    backgroundColor: '#EEEEF0',
    paddingHorizontal: 50,
    paddingTop: 46,
    paddingBottom: 34,
    shadowColor: '#000',
    shadowOpacity: 0.28,
    shadowRadius: 38,
    shadowOffset: { width: 0, height: 30 },
  },
  modalCardDark: {
    backgroundColor: '#151318',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
  },
  closeButton: {
    position: 'absolute',
    right: 32,
    top: 32,
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    color: '#000000',
    fontSize: 30,
    fontWeight: '900',
    textAlign: 'center',
  },
  titleDark: {
    color: '#FFFFFF',
  },
  subtitle: {
    color: '#6E788D',
    fontSize: 16,
    lineHeight: 23,
    textAlign: 'center',
    marginTop: 10,
    marginBottom: 26,
  },
  subtitleDark: {
    color: '#9AA5BD',
  },
  field: {
    gap: 7,
    marginBottom: 15,
  },
  label: {
    color: '#465168',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 1.4,
  },
  labelDark: {
    color: '#8D99B3',
  },
  input: {
    height: 50,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#D7D7DB',
    paddingHorizontal: 20,
    color: '#111111',
    fontSize: 16,
  },
  inputDark: {
    borderColor: 'rgba(255,255,255,0.12)',
    backgroundColor: 'rgba(255,255,255,0.05)',
    color: '#FFFFFF',
  },
  errorText: {
    color: '#EF4444',
    fontSize: 12,
    marginBottom: 8,
  },
  submitButton: {
    height: 52,
    borderRadius: 14,
    backgroundColor: '#000000',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 8,
    shadowColor: '#A020F0',
    shadowOpacity: 0.3,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 16 },
  },
  submitButtonDisabled: {
    opacity: 0.6,
  },
  submitText: {
    color: '#FFFFFF',
    fontWeight: '900',
  },
  registerLink: {
    alignItems: 'center',
    marginTop: 26,
  },
  registerText: {
    color: '#6E788D',
    fontWeight: '900',
  },
  errorTextLight: {
    color: '#DC2626',
  },
});
