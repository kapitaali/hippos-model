const express = require('express');
const fs = require('fs');
const JSONStream = require('JSONStream');
const axios = require('axios');

const app = express();
const port = 3000;
const jsonFilePath = '../racedata/scraped_race_data_2018-2024.json';

// Middleware to parse JSON bodies
app.use(express.json());

// Helper to check if date is in the past
const isPastDate = (dateStr) => {
  const inputDate = new Date(dateStr);
  const today = new Date('2025-03-17'); // Hardcoded for now, could use new Date()
  return inputDate < today;
};

// GET /races/:date
app.get('/races/:date', async (req, res) => {
  const { date } = req.params;

  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return res.status(400).json({ error: 'Invalid date format. Use YYYY-MM-DD' });
  }

  try {
    if (isPastDate(date)) {
      // Past date: read from JSON file
      const stream = fs.createReadStream(jsonFilePath, { encoding: 'utf8' });
      const parser = JSONStream.parse(`*.${date}`);
      let found = false;

      stream.pipe(parser);

      parser.on('data', (data) => {
        if (data) {
          found = true;
          res.json(data);
          stream.destroy(); // Stop reading once we have it
        }
      });

      parser.on('end', () => {
        if (!found) res.status(404).json({ error: `${date} not found in JSON` });
      });

      parser.on('error', (err) => {
        console.error('JSON parse error:', err);
        res.status(500).json({ error: 'Error reading JSON file' });
      });

      stream.on('error', (err) => {
        console.error('File stream error:', err);
        res.status(500).json({ error: 'Error accessing JSON file' });
      });
    } else {
      // Future date: call API
      const currentTrack = 1;
      const apiUrl = `https://heppa.hippos.fi/heppa2_backend/race/${currentDate}/${currentTrack}`; 
      const response = await axios.get(apiUrl);
      res.json(response.data);
    }
  } catch (err) {
    console.error('Request error:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
