// src/context/RefreshContext.tsx
import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { apiService } from '../services/api';
import { DataTelemetry } from '../types/fraudIntelligence';

export interface RefreshContextType {
  isRefreshing: boolean;
  lastUIRefresh: Date | null;
  lastSourceFetch: string | null;
  lastDataProcessing: string | null;
  dataVersion: string | null;
  refreshMessage: string | null;
  refreshMessageType: 'info' | 'success' | 'error' | null;
  refreshError: string | null;
  triggerGlobalRefresh: () => Promise<void>;
  updateTelemetry: (telemetry?: DataTelemetry) => void;
  clearMessage: () => void;
}

const RefreshContext = createContext<RefreshContextType | undefined>(undefined);

export const RefreshProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUIRefresh, setLastUIRefresh] = useState<Date | null>(new Date());
  const [lastSourceFetch, setLastSourceFetch] = useState<string | null>(null);
  const [lastDataProcessing, setLastDataProcessing] = useState<string | null>(null);
  const [dataVersion, setDataVersion] = useState<string | null>(null);
  const [refreshMessage, setRefreshMessage] = useState<string | null>(null);
  const [refreshMessageType, setRefreshMessageType] = useState<'info' | 'success' | 'error' | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  const isRefreshingRef = useRef(false);
  const dataVersionRef = useRef<string | null>(null);
  const dismissTimerRef = useRef<any>(null);

  const clearMessage = useCallback(() => {
    setRefreshMessage(null);
    setRefreshMessageType(null);
    setRefreshError(null);
  }, []);

  const updateTelemetry = useCallback((telemetry?: DataTelemetry) => {
    if (!telemetry) return;
    if (telemetry.last_source_fetch) setLastSourceFetch(telemetry.last_source_fetch);
    if (telemetry.last_data_processing) setLastDataProcessing(telemetry.last_data_processing);
    if (telemetry.data_version) {
      setDataVersion(telemetry.data_version);
      dataVersionRef.current = telemetry.data_version;
    }
  }, []);

  // Fetch initial telemetry on startup
  useEffect(() => {
    let mounted = true;
    apiService
      .getTelemetry(false)
      .then((t) => {
        if (mounted && t) {
          updateTelemetry(t);
        }
      })
      .catch(() => {
        // Non-blocking initial telemetry fetch
      });
    return () => {
      mounted = false;
    };
  }, [updateTelemetry]);

  const triggerGlobalRefresh = useCallback(async () => {
    if (isRefreshingRef.current) return;
    isRefreshingRef.current = true;
    setIsRefreshing(true);
    setRefreshError(null);
    if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);

    const startTime = Date.now();
    const oldVersion = dataVersionRef.current;

    try {
      // 1. Dispatch custom event with forceRefresh=true so active dynamic pages reload data
      window.dispatchEvent(
        new CustomEvent('metaview:refresh', {
          detail: { forceRefresh: true, timestamp: Date.now() },
        })
      );

      // 2. Fetch fresh backend telemetry directly
      const telemetry = await apiService.getTelemetry(true);
      const newVersion = telemetry?.data_version || null;

      // 3. Update telemetry timestamps
      updateTelemetry(telemetry);
      setLastUIRefresh(new Date());

      // 4. Compare version to determine actual freshness status
      if (oldVersion && newVersion && oldVersion === newVersion) {
        setRefreshMessage('تم التحقق من البيانات، ولا توجد تحديثات جديدة');
        setRefreshMessageType('info');
      } else {
        setRefreshMessage('تم تحديث البيانات بنجاح وفق أحدث السجلات المتوفرة');
        setRefreshMessageType('success');
      }

      // Auto-dismiss notification after 5 seconds
      dismissTimerRef.current = setTimeout(() => {
        setRefreshMessage(null);
        setRefreshMessageType(null);
      }, 5000);
    } catch (err: any) {
      console.error('Refresh error:', err);
      const msg = err?.message || 'فشل تحديث البيانات، يرجى المحاولة لاحقاً';
      setRefreshError(msg);
      setRefreshMessage(`تعذر إكمال التحديث: ${msg}`);
      setRefreshMessageType('error');
    } finally {
      // Smooth visual feedback: ensure minimum 400ms duration
      const elapsed = Date.now() - startTime;
      const delay = Math.max(0, 450 - elapsed);
      setTimeout(() => {
        isRefreshingRef.current = false;
        setIsRefreshing(false);
      }, delay);
    }
  }, [updateTelemetry]);

  return (
    <RefreshContext.Provider
      value={{
        isRefreshing,
        lastUIRefresh,
        lastSourceFetch,
        lastDataProcessing,
        dataVersion,
        refreshMessage,
        refreshMessageType,
        refreshError,
        triggerGlobalRefresh,
        updateTelemetry,
        clearMessage,
      }}
    >
      {children}
    </RefreshContext.Provider>
  );
};

export const useRefresh = (): RefreshContextType => {
  const context = useContext(RefreshContext);
  if (!context) {
    throw new Error('useRefresh must be used within a RefreshProvider');
  }
  return context;
};
