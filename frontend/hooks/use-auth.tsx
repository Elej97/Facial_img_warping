import AsyncStorage from '@react-native-async-storage/async-storage';
import { createContext, useContext, useEffect, useMemo, useState } from 'react';

type AuthContextValue = {
  userName: string | null;
  userId: number | null;
  token: string | null;
  isLoading: boolean;
  signIn: (email: string, password: string) => Promise<boolean>;
  signUp: (email: string, username: string, password: string) => Promise<boolean>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

type AuthProviderProps = {
  children: React.ReactNode;
};

const API_BASE = (process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:3000').replace(/\/$/, '');

export function AuthProvider({ children }: AuthProviderProps) {
  const [userName, setUserName] = useState<string | null>(null);
  const [userId, setUserId] = useState<number | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load auth from AsyncStorage on mount
  useEffect(() => {
    const loadAuth = async () => {
      try {
        const savedToken = await AsyncStorage.getItem('authToken');
        const savedUserName = await AsyncStorage.getItem('userName');
        const savedUserId = await AsyncStorage.getItem('userId');

        if (savedToken && savedUserName && savedUserId) {
          setToken(savedToken);
          setUserName(savedUserName);
          setUserId(Number(savedUserId));
        }
      } catch (err) {
        console.error('Auth load error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    loadAuth();
  }, []);

  const signIn = async (email: string, password: string): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (response.ok && data.token) {
        await AsyncStorage.setItem('authToken', data.token);
        await AsyncStorage.setItem('userName', data.username);
        await AsyncStorage.setItem('userId', String(data.userId));

        setToken(data.token);
        setUserName(data.username);
        setUserId(data.userId);
        return true;
      }
      return false;
    } catch (err) {
      console.error('Sign in error:', err);
      return false;
    }
  };

  const signUp = async (email: string, username: string, password: string): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, username, password }),
      });

      const data = await response.json();

      if (response.ok && data.token) {
        await AsyncStorage.setItem('authToken', data.token);
        await AsyncStorage.setItem('userName', data.username);
        await AsyncStorage.setItem('userId', String(data.userId));

        setToken(data.token);
        setUserName(data.username);
        setUserId(data.userId);
        return true;
      }
      return false;
    } catch (err) {
      console.error('Sign up error:', err);
      return false;
    }
  };

  const signOut = async () => {
    try {
      await AsyncStorage.removeItem('authToken');
      await AsyncStorage.removeItem('userName');
      await AsyncStorage.removeItem('userId');

      setToken(null);
      setUserName(null);
      setUserId(null);
    } catch (err) {
      console.error('Sign out error:', err);
    }
  };

  const value = useMemo(
    () => ({
      userName,
      userId,
      token,
      isLoading,
      signIn,
      signUp,
      signOut,
    }),
    [userName, userId, token, isLoading]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }

  return context;
}
