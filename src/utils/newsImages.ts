// src/utils/newsImages.ts
// Helper to provide high-quality, relevant editorial photographs for each fraud news item and incident
// Fallbacks are curated by topic so every single card has a realistic, matching image like in the design mockup.

const TOPIC_IMAGES = {
  // 1. Arrests, police raids, syndicates, handcuffs
  arrest: 'https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=format&fit=crop&w=600&q=80', // legal/handcuffs/justice
  handcuffs: 'https://images.unsplash.com/photo-1589578527966-fdac0f44566c?auto=format&fit=crop&w=600&q=80', // handcuffs

  // 2. Phishing, cyber attack, hacker at laptop in dark cyan lighting
  cyberHacker: 'https://images.unsplash.com/photo-1563986768609-322da13575f3?auto=format&fit=crop&w=600&q=80', // laptop cyber
  phishing: 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=600&q=80', // matrix code / cyber

  // 3. Banking, Central Bank (SAMA), financial investment scams
  banking: 'https://images.unsplash.com/photo-1541354329998-f4d9a9f9297f?auto=format&fit=crop&w=600&q=80', // bank building
  investment: 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=600&q=80', // stock chart / market scam

  // 4. SMS, emails, fake links, message lure
  emailLure: 'https://images.unsplash.com/photo-1596524430615-b46475ddff6e?auto=format&fit=crop&w=600&q=80', // digital message envelope

  // 5. Awareness campaign, cybersecurity workshop
  securityAwareness: 'https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=600&q=80', // cybersecurity shield
  hoodedHacker: 'https://images.unsplash.com/photo-1510511459019-5dda7724fd87?auto=format&fit=crop&w=600&q=80', // cyber security silhouette

  // 6. Generic financial fraud
  financeFraud: 'https://images.unsplash.com/photo-1559526324-4b87b5e36e44?auto=format&fit=crop&w=600&q=80', // financial documents
};

export const getNewsImage = (item: {
  title?: string;
  headline?: string;
  summary?: string | null;
  fraud_category?: string;
  image_url?: string | null;
}): string => {
  // If the article has an explicit safe image URL from the publisher, use it
  if (item.image_url && item.image_url.startsWith('http') && !item.image_url.endsWith('.svg')) {
    return item.image_url;
  }

  const text = `${item.title || ''} ${item.headline || ''} ${item.summary || ''} ${item.fraud_category || ''}`.toLowerCase();

  // Match by keyword
  if (text.includes('شرطة') || text.includes('تطيح') || text.includes('قبض') || text.includes('مقيمين') || text.includes('شبكة') || text.includes('ضبط')) {
    return TOPIC_IMAGES.handcuffs;
  }
  if (text.includes('مدارس') || text.includes('رسوم') || text.includes('إلكتروني') || text.includes('رابط') || text.includes('روابط')) {
    return TOPIC_IMAGES.cyberHacker;
  }
  if (text.includes('ساما') || text.includes('sama') || text.includes('استثمار') || text.includes('تداول') || text.includes('أسهم') || text.includes('بنك')) {
    return TOPIC_IMAGES.investment;
  }
  if (text.includes('رسائل') || text.includes('sms') || text.includes('تصيد') || text.includes('بريد') || text.includes('فخ') || text.includes('واتساب')) {
    return TOPIC_IMAGES.emailLure;
  }
  if (text.includes('توعية') || text.includes('حملة') || text.includes('أمن سيبراني') || text.includes('سيبراني')) {
    return TOPIC_IMAGES.securityAwareness;
  }

  // Default to clean cyber-financial security image
  return TOPIC_IMAGES.financeFraud;
};
