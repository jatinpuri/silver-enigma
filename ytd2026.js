const fs = require('fs');
const { getHistoricalRates } = require('dukascopy-node');

const symbols = ['eurusd','gbpusd','audusd','usdchf','usdjpy','eurgbp','xauusd'];

(async () => {
  for (const s of symbols) {
    const data = await getHistoricalRates({
      instrument: s,
      dates: { from: new Date('2025-11-01T00:00:00Z'), to: new Date('2026-08-09T00:00:00Z') },
      timeframe: 'h1',
      priceType: 'bid',
      format: 'array',
      utcOffset: 0,
      volumes: false,
      ignoreFlats: true,
      batchSize: 10,
      pauseBetweenBatchesMs: 250,
      retryCount: 3,
      retryOnEmpty: true,
      failAfterRetryCount: false,
      pauseBetweenRetriesMs: 500,
    });
    fs.writeFileSync(`ytd_${s.toUpperCase()}_h1.json`, JSON.stringify(data));
    console.log(s, data.length);
  }
})().catch(e => { console.error(e); process.exit(1); });
