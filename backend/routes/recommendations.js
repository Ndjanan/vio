// Express route: /api/recommendations
// Requires: node-fetch or built-in fetch (Node 18+), and supabaseClient.js in ../integration
const express = require('express');
const router = express.Router();
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
const supabase = require('../integration/supabaseClient');

// FASTAPI_URL env variable should point to the running FastAPI service (e.g., http://localhost:8000)
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

router.post('/', async (req, res) => {
  try {
    const user_id = req.body.user_id;
    const limit = req.body.limit || 10;
    // Call FastAPI /predict with user_id
    const r = await fetch(`${FASTAPI_URL}/predict`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({user_id, limit})
    });
    if (!r.ok) {
      const txt = await r.text();
      return res.status(500).send({error: 'FastAPI error', detail: txt});
    }
    const data = await r.json();
    // Optionally, you can write/update recommendations locally via Supabase client here
    return res.json({recommendations: data});
  } catch (err) {
    console.error(err);
    res.status(500).send({error: err.message});
  }
});

module.exports = router;
