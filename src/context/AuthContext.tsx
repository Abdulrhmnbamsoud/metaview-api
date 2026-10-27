// src/context/AuthContext.tsx
import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { User, UserRole } from '../types/auth';
import { apiService, getSessionToken, setSessionToken } from '../services/api';

interface AuthContextType {
  user: User | null;
  role: UserRole | null;
  permissions: string[];
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  can: (permissionKey: string) => boolean;
  isRole: (roles: UserRole | UserRole[]) => boolean;
  switchPersona: (targetRole: UserRole) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Preset test credentials for instant persona switching
const TEST_ACCOUNTS: Record<UserRole, { email: string; pass: string; name: string }> = {
  admin: {
    email: 'admin@metaview.local',
    pass: 'Admin#MetaView2026!',
    name: 'مدير المنصة (Admin)',
  },
  analyst: {
    email: 'analyst@metaview.sa',
    pass: 'Analyst#Secret2026!',
    name: 'سارة العتيبي (محلل احتيال أول)',
  },
  viewer: {
    email: 'viewer@metaview.sa',
    pass: 'Viewer#Secret2026!',
    name: 'عبدالله السعيد (مستعرض تنفيذي)',
  },
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Initialize session on mount
  useEffect(() => {
    let mounted = true;
    const initAuth = async () => {
      const token = getSessionToken();
      if (token) {
        try {
          const profile = await apiService.getMe();
          if (mounted) {
            setUser(profile);
            setLoading(false);
            return;
          }
        } catch {
          // Token expired or invalid
          setSessionToken(null);
        }
      }

      // Default state: Public Viewer (unauthenticated read-only)
      if (mounted) {
        setUser(null);
        setLoading(false);
      }
    };

    initAuth();
    return () => {
      mounted = false;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    setLoading(true);
    try {
      const res = await apiService.login({ email, password });
      setUser(res.user);
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await apiService.logout();
    } catch {
      // ignore
    } finally {
      setUser(null);
      setSessionToken(null);
    }
  }, []);

  const can = useCallback(
    (permissionKey: string): boolean => {
      if (!user) return false;
      return user.permissions.includes(permissionKey);
    },
    [user]
  );

  const isRole = useCallback(
    (roles: UserRole | UserRole[]): boolean => {
      if (!user) return false;
      if (Array.isArray(roles)) {
        return roles.includes(user.role);
      }
      return user.role === roles;
    },
    [user]
  );

  // Fast persona switcher for evaluator demonstration
  const switchPersona = useCallback(
    async (targetRole: UserRole) => {
      setLoading(true);
      try {
        const creds = TEST_ACCOUNTS[targetRole];
        try {
          const res = await apiService.login({ email: creds.email, password: creds.pass });
          setUser(res.user);
        } catch {
          // If the account hasn't been invited or created yet, create it via admin invite & accept flow!
          if (user?.role !== 'admin') {
            // temporarily login as admin to invite
            const adminRes = await apiService.login({
              email: TEST_ACCOUNTS.admin.email,
              password: TEST_ACCOUNTS.admin.pass,
            });
            setUser(adminRes.user);
          }

          // Invite member
          const invRes = await apiService.inviteMember({
            email: creds.email,
            role: targetRole,
          });

          // Accept invite
          const acceptRes = await apiService.acceptInvitation({
            token: invRes.invite_token,
            full_name: creds.name,
            password: creds.pass,
          });
          setUser(acceptRes.user);
        }
      } catch (err: any) {
        console.error('Failed to switch persona:', err);
      } finally {
        setLoading(false);
      }
    },
    [user]
  );

  return (
    <AuthContext.Provider
      value={{
        user,
        role: user?.role || null,
        permissions: user?.permissions || [],
        loading,
        login,
        logout,
        can,
        isRole,
        switchPersona,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
