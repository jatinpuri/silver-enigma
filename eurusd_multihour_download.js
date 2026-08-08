const fs = require('fs');
const { getHistoricalRates } = require('dukascopy-node');

(async () => {
  const data = await getHistoricalRates({
    instrument: 'eurusd',
    dates: { from: new Date('2011-10-01T00:00:00Z'), to: new Date('2026-08-09T00:00:00Z') },
    timeframe: 'h1',
    priceType: 'bid',
    format: 'array',
    utcOffset: 0,
    volumes: false,
    ignoreFlats: true,
    batchSize: 50,
    pauseBetweenBatchesMs: 100,
    retryCount: 3,
    retryOnEmpty: true,
    failAfterRetryCount: false,
    pauseBetweenRetriesMs: 500,
  });
  fs.writeFileSync('eurusd_h1_2011_2026.json', JSON.stringify(data));
  console.log('EURUSD bars', data.length);
})().catch(e => { console.error(e); process.exit(1); });
