-- CreateTable: Raw observation records ingested from CSV
CREATE TABLE "observations" (
    "id" SERIAL NOT NULL,
    "hcp_id" TEXT NOT NULL,
    "month" TIMESTAMP(3) NOT NULL,
    "suggestion_type" TEXT NOT NULL,
    "action_type" TEXT NOT NULL,
    "outcome_type" TEXT NOT NULL,
    "suggestion_count" DOUBLE PRECISION NOT NULL,
    "action_count" DOUBLE PRECISION NOT NULL,
    "outcome_count" DOUBLE PRECISION NOT NULL,
    "specialty_code" TEXT,
    "region_code" TEXT,
    "tenure_months" DOUBLE PRECISION,
    "prior_trx" DOUBLE PRECISION,
    "prior_nbrx" DOUBLE PRECISION,
    "total_suggestions" DOUBLE PRECISION,
    "theme" TEXT,
    "upload_batch_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "observations_pkey" PRIMARY KEY ("id")
);

-- CreateTable: Model metadata storing trained coefficients
CREATE TABLE "model_metadata" (
    "id" SERIAL NOT NULL,
    "model_name" TEXT NOT NULL,
    "model_type" TEXT NOT NULL,
    "suggestion_type" TEXT NOT NULL,
    "action_type" TEXT NOT NULL,
    "outcome_type" TEXT,
    "feature_names" TEXT[],
    "coefficients" JSONB NOT NULL,
    "intercept" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "scaler_means" JSONB,
    "scaler_stds" JSONB,
    "categorical_map" JSONB,
    "trained_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "run_id" TEXT NOT NULL,

    CONSTRAINT "model_metadata_pkey" PRIMARY KEY ("id")
);

-- CreateTable: Model run history
CREATE TABLE "model_runs" (
    "id" SERIAL NOT NULL,
    "run_id" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "suggestion_type" TEXT NOT NULL,
    "action_type" TEXT NOT NULL,
    "outcome_type" TEXT NOT NULL,
    "adstock_enabled" BOOLEAN NOT NULL DEFAULT false,
    "adstock_decay" DOUBLE PRECISION,
    "theme" TEXT,
    "total_observations" INTEGER,
    "r_squared_path_a" DOUBLE PRECISION,
    "r_squared_path_b" DOUBLE PRECISION,
    "mae_path_a" DOUBLE PRECISION,
    "mae_path_b" DOUBLE PRECISION,
    "started_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completed_at" TIMESTAMP(3),
    "error_message" TEXT,
    "model_metadata_id" INTEGER,

    CONSTRAINT "model_runs_pkey" PRIMARY KEY ("id")
);

-- CreateTable: Per-HCP-month lift results
CREATE TABLE "lift_results" (
    "id" SERIAL NOT NULL,
    "run_id" TEXT NOT NULL,
    "hcp_id" TEXT NOT NULL,
    "month" TIMESTAMP(3) NOT NULL,
    "suggestion_type" TEXT NOT NULL,
    "action_type" TEXT NOT NULL,
    "outcome_type" TEXT NOT NULL,
    "theme" TEXT,
    "predicted_action" DOUBLE PRECISION NOT NULL,
    "counterfactual_action" DOUBLE PRECISION NOT NULL,
    "incremental_action" DOUBLE PRECISION NOT NULL,
    "predicted_outcome" DOUBLE PRECISION NOT NULL,
    "counterfactual_outcome" DOUBLE PRECISION NOT NULL,
    "incremental_outcome" DOUBLE PRECISION NOT NULL,
    "observed_action" DOUBLE PRECISION NOT NULL,
    "observed_outcome" DOUBLE PRECISION NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "lift_results_pkey" PRIMARY KEY ("id")
);

-- CreateTable: Feature importance rankings
CREATE TABLE "feature_importance" (
    "id" SERIAL NOT NULL,
    "model_metadata_id" INTEGER NOT NULL,
    "feature_name" TEXT NOT NULL,
    "importance" DOUBLE PRECISION NOT NULL,
    "rank" INTEGER NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "feature_importance_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "model_runs_run_id_key" ON "model_runs"("run_id");

-- AddForeignKey
ALTER TABLE "model_runs" ADD CONSTRAINT "model_runs_model_metadata_id_fkey" FOREIGN KEY ("model_metadata_id") REFERENCES "model_metadata"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "lift_results" ADD CONSTRAINT "lift_results_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "model_runs"("run_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "feature_importance" ADD CONSTRAINT "feature_importance_model_metadata_id_fkey" FOREIGN KEY ("model_metadata_id") REFERENCES "model_metadata"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- Create indexes for common query patterns
CREATE INDEX "observations_suggestion_type_idx" ON "observations"("suggestion_type");
CREATE INDEX "observations_action_type_idx" ON "observations"("action_type");
CREATE INDEX "observations_outcome_type_idx" ON "observations"("outcome_type");
CREATE INDEX "observations_month_idx" ON "observations"("month");
CREATE INDEX "observations_hcp_id_idx" ON "observations"("hcp_id");
CREATE INDEX "lift_results_run_id_idx" ON "lift_results"("run_id");
CREATE INDEX "lift_results_month_idx" ON "lift_results"("month");
