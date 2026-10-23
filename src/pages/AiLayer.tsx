// src/pages/AiLayer.tsx
import React from 'react';
import {
  Sparkles,
  Cpu,
  Search,
  Layers,
  HeartPulse,
  Languages,
  Network,
  GitCompare,
  FileText,
  ShieldCheck,
  CheckCircle2,
  Clock,
  Terminal,
} from 'lucide-react';

export const AiLayer: React.FC = () => {
  const capabilities = [
    {
      title: 'البحث الدلالي (Semantic Search)',
      status: 'معمارية جاهزة للتكامل',
      statusType: 'ready',
      desc: 'البحث عن الأحداث والتطورات الإخبارية حسب المعنى والارتباط المفاهيمي، وليس فقط مطابقة الكلمات الحرفية.',
      endpoint: 'POST /ai/semantic-search',
      payload: '{"query": string, "vector_top_k": 20}',
      icon: Search,
    },
    {
      title: 'تجميع الأحداث المشتركة (Event Clustering)',
      status: 'معمارية جاهزة للتكامل',
      statusType: 'ready',
      desc: 'ربط ودمج الأخبار الصادرة عن مصادر متعددة والتي تتناول نفس الحدث الواقعي في مجموعة حدثية موحدة (Story Cluster).',
      endpoint: 'POST /ai/cluster-events',
      payload: '{"time_window_hours": 24, "similarity_threshold": 0.85}',
      icon: Network,
    },
    {
      title: 'التصنيف الآلي متعدد الأبعاد (Topic Classification)',
      status: 'مخطط للتطوير',
      statusType: 'planned',
      desc: 'تصنيف المقالات تلقائياً إلى قطاعات استراتيجية: سياسة دولية، اقتصاد، أمن وطاقة، ذكاء اصطناعي وتكنولوجيا.',
      endpoint: 'POST /ai/classify',
      payload: '{"article_ids": number[]}',
      icon: Layers,
    },
    {
      title: 'تحليل النبرة التحريرية (Sentiment & Tone)',
      status: 'مخطط للتطوير',
      statusType: 'planned',
      desc: 'قياس النبرة اللغوية للمقال، مع التأكيد المنهجي الصارم على أن قياس المشاعر اللغوية لا يمثل حكماً على صدقية أو كذب الخبر.',
      endpoint: 'POST /ai/sentiment',
      payload: '{"text": string, "type": "headline" | "summary"}',
      icon: HeartPulse,
    },
    {
      title: 'الاستخبارات اللغوية والترجمة الفورية (Multilingual Intelligence)',
      status: 'مخطط للتطوير',
      statusType: 'planned',
      desc: 'ترجمة فورية لعناوين وملخصات وكالات الأنباء الدولية (مثل لوموند ودويتشه فيله) إلى العربية مع الاحتفاظ بالنص الأصلي كاملاً.',
      endpoint: 'POST /ai/translate',
      payload: '{"target_lang": "ar", "source_text": string}',
      icon: Languages,
    },
    {
      title: 'مقارنة التغطيات بين المصادر (Cross-Source Comparison)',
      status: 'مخطط للتطوير',
      statusType: 'planned',
      desc: 'رصد الفروقات والتركيز التحريري بين تغطية مصادر مختلفة لنفس الحدث (مثل رويترز مقابل سي إن إن).',
      endpoint: 'POST /ai/compare-coverage',
      payload: '{"article_url_1": string, "article_url_2": string}',
      icon: GitCompare,
    },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-white/5">
        <div>
          <h1 className="text-2xl font-black text-white royal-gradient-text">
            طبقة الذكاء الاصطناعي والاستخبارات المتقدمة (AI Intelligence Layer)
          </h1>
          <p className="text-xs text-gray-400 mt-1">
            البنية المعمارية المخططة لخدمات التحليل الذكي المستقلة
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[#4cd7f6]/10 border border-[#4cd7f6]/30 text-[#acedff] text-xs font-mono">
          <Cpu className="w-4 h-4 text-[#4cd7f6]" />
          <span>MICROSERVICES ARCHITECTURE</span>
        </div>
      </div>

      {/* Transparency & Integrity Box */}
      <div className="p-5 rounded-2xl bg-[#0F172A] border border-[#4cd7f6]/20 flex items-start gap-4">
        <ShieldCheck className="w-6 h-6 text-[#4cd7f6] shrink-0 mt-0.5" />
        <div className="text-xs text-gray-300 leading-relaxed space-y-1">
          <strong className="text-white text-sm block">
            معيار النزاهة والشفافية التقنية (Rule #12 & Rule #15):
          </strong>
          <p>
            تلتزم MetaView 3 بعدم اختلاق أو توليد أي تحليلات ذكاء اصطناعي وهمية داخل الواجهة. خدمات الذكاء الاصطناعي مصممة كطبقة خدمات مستقلة (Microservices) يتم ربطها بنقاط اتصال مخصصة في Backend واختبارها بدقة قبل إعلان تفعيلها للعموم.
          </p>
        </div>
      </div>

      {/* Capabilities Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {capabilities.map((cap, idx) => {
          const Icon = cap.icon;
          return (
            <div
              key={idx}
              className="sovereign-card rounded-2xl p-6 border border-white/10 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 rounded-xl bg-[#4cd7f6]/10 border border-[#4cd7f6]/20 flex items-center justify-center text-[#4cd7f6]">
                    <Icon className="w-5 h-5" />
                  </div>
                  <span
                    className={`px-2.5 py-1 rounded-lg text-[10px] font-mono font-bold ${
                      cap.statusType === 'ready'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                        : 'bg-white/5 text-gray-400 border border-white/10'
                    }`}
                  >
                    {cap.status}
                  </span>
                </div>

                <h3 className="text-base font-bold text-white mb-2">{cap.title}</h3>
                <p className="text-xs text-gray-400 leading-relaxed mb-4">{cap.desc}</p>
              </div>

              <div className="pt-4 border-t border-white/5 space-y-2">
                <div className="flex items-center gap-1.5 text-[11px] font-mono text-gray-400">
                  <Terminal className="w-3.5 h-3.5 text-[#4cd7f6]" />
                  <span>{cap.endpoint}</span>
                </div>
                <div className="p-2 rounded-lg bg-black/60 border border-white/5 text-[10px] font-mono text-gray-500 truncate">
                  {cap.payload}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
