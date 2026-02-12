/**
 * API Routes
 *
 * POST /upload-data         → Accept CSV, store in database
 * POST /train-models        → Train Path A and Path B models
 * POST /run-lift            → Execute full counterfactual simulation
 * GET  /summary             → Return monthly aggregated results
 * GET  /feature-importance  → Return ranked features
 * GET  /health              → Health check
 */

import { Router } from 'express';
import multer from 'multer';
import { uploadData } from '../controllers/uploadController';
import { trainModels } from '../controllers/trainController';
import { runLift } from '../controllers/liftController';
import { getSummary } from '../controllers/summaryController';
import { getFeatureImportance } from '../controllers/featureImportanceController';

const router = Router();

// Multer configured for in-memory CSV upload
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB limit
  fileFilter: (_req, file, cb) => {
    if (
      file.mimetype === 'text/csv' ||
      file.originalname.endsWith('.csv')
    ) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'));
    }
  },
});

// Health check
router.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// POST /upload-data — Upload CSV observations
router.post('/upload-data', upload.single('file'), uploadData);

// POST /train-models — Train Path A and Path B models
router.post('/train-models', trainModels);

// POST /run-lift — Execute full counterfactual simulation
router.post('/run-lift', runLift);

// GET /summary — Monthly aggregated results
router.get('/summary', getSummary);

// GET /feature-importance — Ranked feature importance
router.get('/feature-importance', getFeatureImportance);

export default router;
