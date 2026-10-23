// src/pages/SystemHealth.tsx
// Deployment and scheduler diagnostics from /saudi-fraud/runtime (no secrets are exposed by the API).
import React, { useCallback, useEffect, useState } from 'react';
import { HeartPulse, HardDrive, Timer, KeyRound } from 'lucide-react';
import { apiService } from '../services/api';
import { SaudiRuntime } from '../types/saudi';
import { HealthStatus } from '../types';
import { Panel, KpiTile, Chip, LoadingBlock, ErrorBlock, PageHeader, fmtDateTime } from '../components/ui';

const fmtBytes = (n?: number | null) => {
  if (n == null) return '—';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i ? 1 : 0)} ${u[i]}`;
};

const Row: React.FC<{ k: string; v: React.ReactNode }> = ({ k, v }) => (
  <div className="flex items-center justify-between gap-3 py-2 px-3 rounded-lg bg-surface-container">
    <span className="text-xs text-ink-outline">{k}</span>
    <span className="text-[13px] text-ink num-mono text-left break-all">{v}</span>
  </div>
);

export const SystemHealth: React.FC = () => {
  const [rt, setRt] = useState<SaudiRuntime | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [r, h] = await Promise.all([apiService.getSaudiRuntime(), apiService.getHealth()]);
      setRt(r);
      setHealth(h);
    } catch (e: any) {
      setError(e?.message || 'خطأ غير معروف');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, [load]);

  if (loading && !rt) return <LoadingBlock label="جارٍ فحص النظام…" />;
  if (error && !rt) return <ErrorBlock message={error} onRetry={load} />;
  if (!rt) return null;

  const online = health && health.status !== 'offline';
  const d = rt.disk;

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="صحة النظام (System Health)"
        title="تشخيص النشر والجدولة"
        subtitle="قراءة فقط من /saudi-fraud/runtime. لا تُعرض أي أسرار أو رموز وصول."
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiTile label="الواجهة البرمجية" value={online ? 'متصلة' : 'غير متصلة'} tone={online ? 'low' : 'critical'} icon={<HeartPulse className="w-4 h-4" />} hint={health ? `${health.rows.toLocaleString()} خبر مخزّن` : undefined} />
        <KpiTile label="الجدولة" value={rt.scheduler.thread_alive ? 'تعمل' : 'متوقفة'} tone={rt.scheduler.thread_alive ? 'low' : 'medium'} icon={<Timer className="w-4 h-4" />} hint={rt.scheduler.last_decision ? `القرار الأخير: ${rt.scheduler.last_decision}` : undefined} />
        <KpiTile label="قاعدة البيانات" value={fmtBytes(d.db_size_bytes)} tone="brand" icon={<HardDrive className="w-4 h-4" />} hint={`وضع السجل: ${rt.sqlite_journal_mode ?? '—'}`} />
        <KpiTile label="الإدارة" value={rt.admin_token_configured ? 'مُفعّلة' : 'غير مُهيّأة'} tone={rt.admin_token_configured ? 'low' : 'medium'} icon={<KeyRound className="w-4 h-4" />} hint="رمز إدارة قرارات الدمج" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="النشر">
          <div className="space-y-2">
            <Row k="الخدمة" v={rt.service || '—'} />
            <Row k="الفرع" v={rt.branch || '—'} />
            <Row k="الإصدار (commit)" v={rt.commit ? rt.commit.slice(0, 12) : 'غير مُعلن (بيئة محلية)'} />
            <Row k="على Render" v={rt.on_render ? 'نعم' : 'لا'} />
            <Row k="مهلة التحديث اليدوي" v={`${rt.manual_cooldown_seconds} ثانية`} />
          </div>
        </Panel>

        <Panel title="التخزين">
          <div className="space-y-2">
            <Row k="مسار القاعدة" v={d.db_path} />
            <Row k="داخل مجلد البيانات" v={d.db_inside_data_dir ? 'نعم' : 'لا'} />
            <Row k="قرص دائم منفصل" v={d.data_dir_on_separate_device == null ? '—' : d.data_dir_on_separate_device ? 'نعم' : 'لا'} />
            <Row k="المساحة الحرة" v={fmtBytes(d.free_bytes)} />
            {d.error && <Row k="خطأ" v={d.error} />}
          </div>
        </Panel>

        <Panel title="الجدولة والاستيعاب">
          <div className="space-y-2">
            <Row k="آخر نبضة" v={fmtDateTime(rt.scheduler.last_tick_at)} />
            <Row k="آخر بدء تشغيل" v={fmtDateTime(rt.scheduler.last_run_started_at)} />
            <Row k="القرار الأخير" v={rt.scheduler.last_decision || '—'} />
            <Row k="آخر خطأ" v={rt.scheduler.last_error || 'لا يوجد'} />
          </div>
        </Panel>

        <Panel title="قفل الاستيعاب (Lease)">
          <div className="flex items-center gap-2">
            <Chip tone={rt.lease.active ? 'medium' : 'low'}>{rt.lease.active ? 'استيعاب جارٍ' : 'خامل'}</Chip>
          </div>
          {rt.lease.active && (
            <div className="space-y-2">
              <Row k="المالك" v={rt.lease.holder || '—'} />
              <Row k="بدأ" v={fmtDateTime(rt.lease.acquired_at)} />
              <Row k="ينتهي" v={fmtDateTime(rt.lease.expires_at)} />
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
};

export default SystemHealth;
