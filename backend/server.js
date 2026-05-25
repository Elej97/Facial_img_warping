require('dotenv').config();
const express = require('express');
const cors = require('cors');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const path = require('path');
const db = require('./db');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://127.0.0.1:8000';
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';

const allowedWarpModes = new Set(['smile', 'eyebrow', 'lip', 'slim']);
const allowedModels = new Set(['MediaPipe', 'Dlib', 'DeepFace']);

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(
  '/api/assets/accessories',
  express.static(path.join(__dirname, '..', 'python_service', 'assets', 'accessories'))
);
app.use((req, _res, next) => {
  req._ts = Date.now();
  next();
});

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const respond = (res, data, meta = {}) =>
  res.json({
    ok: true,
    source: 'PostgreSQL',
    timestamp: db.nowIso(),
    ...data,
    meta,
  });

const fail = (res, status, message, details) =>
  res.status(status).json({
    ok: false,
    source: 'PostgreSQL',
    timestamp: db.nowIso(),
    error: message,
    details,
  });

const cvUnavailable = (res) =>
  res.status(503).json({
    success: false,
    message: 'CV service unavailable - is the Python service running?',
  });

async function proxyBinaryToPython(path, req, res) {
  try {
    const response = await fetch(`${PYTHON_SERVICE_URL}${path}`, {
      method: 'POST',
      headers: req.headers['content-type']
        ? { 'content-type': req.headers['content-type'] }
        : {},
      body: req,
      duplex: 'half',
    });

    const contentType = response.headers.get('content-type') || 'application/octet-stream';
    const disposition = response.headers.get('content-disposition');

    if (contentType.includes('application/json')) {
      const json = await response.json();
      return res.status(response.status).json(json);
    }

    const buffer = Buffer.from(await response.arrayBuffer());
    res.status(response.status);
    res.setHeader('content-type', contentType);
    if (disposition) res.setHeader('content-disposition', disposition);
    return res.send(buffer);
  } catch (err) {
    if (err.code === 'ECONNREFUSED' || err.cause?.code === 'ECONNREFUSED') {
      return cvUnavailable(res);
    }
    return res.status(500).json({ success: false, message: err.message });
  }
}

// Forward multipart FormData to the Python service and return its JSON response.
async function proxyToPython(path, req, res) {
  try {
    const response = await fetch(`${PYTHON_SERVICE_URL}${path}`, {
      method: 'POST',
      headers: req.headers['content-type']
        ? { 'content-type': req.headers['content-type'] }
        : {},
      body: req,
      duplex: 'half',
    });
    const json = await response.json();
    return res.status(response.status).json(json);
  } catch (err) {
    if (err.code === 'ECONNREFUSED' || err.cause?.code === 'ECONNREFUSED') {
      return cvUnavailable(res);
    }
    return res.status(500).json({ success: false, message: err.message });
  }
}

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return fail(res, 401, 'Token bulunamadi.');
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return fail(res, 403, 'Gecersiz token.');
    }
    req.userId = user.id;
    req.userEmail = user.email;
    next();
  });
};

// Health check
app.get('/api/health', async (req, res) => {
  try {
    const result = await pool.query('SELECT NOW()');
    res.json({ 
      status: 'ok', 
      database: 'connected',
      time: result.rows[0].now 
    });
  } catch (err) {
    console.error('[HEALTH] Database error:', err.message);
    res.status(503).json({ 
      status: 'error', 
      database: 'disconnected',
      error: err.message 
    });
  }
});

// Auth endpoints
app.post('/api/auth/register', async (req, res) => {
  try {
    const { email, username, password } = req.body;

    if (!email || !username || !password) {
      return fail(res, 400, 'Email, username ve password gerekli.');
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const user = await db.createUser({
      email: email.toLowerCase(),
      username: username.trim(),
      password: hashedPassword,
    });

    if (!user) {
      console.error(`[REGISTER] User creation failed for email: ${email}`);
      return fail(res, 409, 'Email veya username zaten kayitli.');
    }

    console.log(`[REGISTER] New user created: ${user.id} - ${user.email}`);

    const token = jwt.sign(
      { id: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: '7d' }
    );

    return respond(res, {
      userId: user.id,
      email: user.email,
      username: user.username,
      token,
    });
  } catch (err) {
    console.error('[REGISTER] Error:', err.message);
    return fail(res, 500, 'Kayit hatasi.', { message: err.message });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return fail(res, 400, 'Email ve password gerekli.');
    }

    const user = await db.getUserByEmail(email.toLowerCase());
    if (!user) {
      return fail(res, 401, 'Email veya password yanlis.');
    }

    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return fail(res, 401, 'Email veya password yanlis.');
    }

    const token = jwt.sign(
      { id: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: '7d' }
    );

    return respond(res, {
      userId: user.id,
      email: user.email,
      username: user.username,
      token,
    });
  } catch (err) {
    return fail(res, 500, 'Giris hatasi.', { message: err.message });
  }
});

app.get('/api/auth/me', authenticateToken, async (req, res) => {
  try {
    const user = await db.getUserById(req.userId);
    if (!user) {
      return fail(res, 404, 'Kullanici bulunamadi.');
    }

    return respond(res, {
      userId: user.id,
      email: user.email,
      username: user.username,
    });
  } catch (err) {
    return fail(res, 500, 'Hata.', { message: err.message });
  }
});

app.get('/api/auth/library', authenticateToken, async (req, res) => {
  try {
    const uploads = await db.getUploadsByUserId(req.userId);
    return respond(res, {
      userId: req.userId,
      uploads,
    });
  } catch (err) {
    return fail(res, 500, 'Kütüphane yükleme hatasi.', { message: err.message });
  }
});

app.get('/api/health', async (_req, res) => {
  try {
    const counts = await db.getCounts();
    return respond(
      res,
      {
        service: 'facial-warping-backend',
        port: Number(PORT),
        pythonService: PYTHON_SERVICE_URL,
        database: process.env.DATABASE_URL?.replace(/:.*@/, ':***@'),
        modules: ['preprocess', 'landmarks', 'expression-transfer', 'warp', 'frequency', 'ai-aging', 'evaluation', 'export'],
        counts,
      },
      { phase: 'postgres-ready' }
    );
  } catch (err) {
    return fail(res, 503, 'Database connection failed.', { message: err.message });
  }
});

// Proxy multipart image endpoints directly to the Python CV service.
app.post('/api/preprocess', (req, res) => proxyToPython('/preprocess', req, res));
app.post('/api/estimate-age', (req, res) => proxyToPython('/estimate-age', req, res));
app.post('/api/aging/ai', (req, res) => proxyToPython('/aging/ai', req, res));
app.post('/api/aging/compare', (req, res) => proxyToPython('/aging/compare', req, res));
app.post('/api/landmarks', (req, res) => proxyToPython('/landmarks', req, res));
app.post('/api/expression/transfer', (req, res) => proxyToPython('/expression/transfer', req, res));
app.post('/api/warp', (req, res) => proxyToPython('/warp', req, res));
app.post('/api/warp/pro', (req, res) => proxyToPython('/warp/pro', req, res));
app.post('/api/frequency', (req, res) => proxyToPython('/frequency', req, res));
app.post('/api/frequency/pro', (req, res) => proxyToPython('/frequency/pro', req, res));
app.post('/api/pro/apply-makeup', (req, res) => proxyToPython('/pro/apply-makeup', req, res));
app.post('/api/pro/apply-accessory', (req, res) => proxyToPython('/pro/apply-accessory', req, res));
app.post('/api/export/csv', (req, res) => proxyBinaryToPython('/export/csv', req, res));
app.post('/api/export/pdf', (req, res) => proxyBinaryToPython('/export/pdf', req, res));
app.post('/api/report/export', (req, res) => proxyToPython('/report/export', req, res));

app.post('/api/evaluation', async (req, res) => {
  const upload = await resolveUploadOrFail(res, req.body);
  if (!upload) return;

  const exp = clamp(Number(req.body?.expressionIntensity) || 0, 0, 100);
  const age = clamp(Number(req.body?.agingLevel) || 0, 0, 100);

  const mse = Number((80 + exp * 0.75 + age * 0.65).toFixed(2));
  const psnr = Number((42 - mse / 18).toFixed(2));
  const ssim = Number(Math.max(0.55, 0.97 - mse / 450).toFixed(3));

  const evaluationRun = await db.createEvaluationRun({ uploadId: upload.id, mse, psnr, ssim });

  return respond(
    res,
    {
      uploadId: upload.id,
      evaluationRunId: evaluationRun.id,
      metrics: { mse, psnr, ssim },
      thresholds: { psnrGoodAbove: 30, ssimGoodAbove: 0.8 },
      quality: {
        psnr: psnr >= 30 ? 'good' : 'needs-improvement',
        ssim: ssim >= 0.8 ? 'good' : 'needs-improvement',
      },
    },
    { stage: 'evaluation' }
  );
});

app.post('/api/export', async (req, res) => {
  const upload = await resolveUploadOrFail(res, req.body);
  if (!upload) return;

  const { format = 'CSV', target = 'results' } = req.body || {};
  const fmt = String(format).toUpperCase();
  if (fmt !== 'CSV' && fmt !== 'PDF') {
    return fail(res, 400, 'Gecersiz format.', { allowed: ['CSV', 'PDF'], received: format });
  }

  const exportRun = await db.createExportRun({
    uploadId: upload.id,
    format: fmt,
    target,
    fileName: `${target}-${Date.now()}.${fmt.toLowerCase()}`,
  });

  return respond(
    res,
    {
      uploadId: upload.id,
      exportRunId: exportRun.id,
      format: fmt,
      target,
      fileName: exportRun.file_name,
      message: `${target} icin ${fmt} export hazir (database).`,
    },
    { stage: 'export' }
  );
});

const resolveUploadOrFail = async (res, body) => {
  const upload = await db.resolveUpload(body || {});
  if (!upload) {
    fail(res, 400, 'Once bir gorsel yuklenmelidir.');
    return null;
  }
  return upload;
};

app.use((err, _req, res, _next) =>
  fail(res, 500, 'Beklenmeyen sunucu hatasi.', { message: err?.message || 'unknown' })
);

app.listen(PORT, () => {
  console.log(`Facial backend listening on http://localhost:${PORT}`);
  console.log(`Proxying CV requests to ${PYTHON_SERVICE_URL}`);
});
