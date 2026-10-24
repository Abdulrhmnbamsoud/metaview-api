// src/utils/theme.ts
export type Theme = 'light' | 'dark';

export const getStoredTheme = (): Theme => {
  if (typeof window === 'undefined') return 'light';
  try {
    const saved = localStorage.getItem('metaview.theme');
    return saved === 'dark' ? 'dark' : 'light';
  } catch {
    return 'light';
  }
};

export const applyTheme = (theme: Theme) => {
  if (typeof document === 'undefined') return;
  try {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('metaview.theme', theme);
  } catch (e) {
    console.error('Error applying theme:', e);
  }
};
