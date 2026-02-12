/**
 * Causal Lift Web Platform — Server Entry Point
 *
 * Express server providing the API for:
 *   - CSV data upload
 *   - Model training (Path A: Suggestion→Action, Path B: Action→Outcome)
 *   - Counterfactual lift simulation
 *   - Summary and feature importance reporting
 */

import express from 'express';
import cors from 'cors';
import routes from './routes';

const app = express();
const PORT = process.env.PORT || 4000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// API routes
app.use('/api', routes);

// Global error handler
app.use(
  (
    err: Error,
    _req: express.Request,
    res: express.Response,
    _next: express.NextFunction
  ) => {
    console.error('Unhandled error:', err.message);
    res.status(500).json({ error: err.message || 'Internal server error' });
  }
);

app.listen(PORT, () => {
  console.log(`Causal Lift Web Platform server running on port ${PORT}`);
});

export default app;
