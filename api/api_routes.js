const express = require('express');
const multer = require('multer');
const { body, param, query, validationResult } = require('express-validator');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const router = express.Router();

// Configure multer for file uploads
const upload = multer({
    dest: 'uploads/',
    limits: {
        fileSize: 50 * 1024 * 1024 // 50MB limit or we can do more if needed I'm not sure
    },
    fileFilter: (req, file, cb) => {
        const allowedTypes = [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
            'application/vnd.ms-excel', // .xls
            'text/csv', // .csv
            'application/json' // .json
        ];
        
        if (allowedTypes.includes(file.mimetype) || file.originalname.match(/\.(xlsx|xls|csv|json)$/)) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only Excel, CSV, and JSON files are allowed.'), false);
        }
    }
});

// Validation middleware
const validateRequest = (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({
            error: {
                message: 'Validation failed',
                details: errors.array()
            }
        });
    }
    next();
};

// Excel Processing Endpoints
// Upload and load Excel file
router.post('/excel/upload',
    upload.single('file'),
    async (req, res, next) => {
        try {
            if (!req.file) {
                return res.status(400).json({
                    error: {
                        message: 'No file uploaded',
                        status: 400
                    }
                });
            }

            const filePath = req.file.path;
            const originalName = req.file.originalname;
            
            // Process file with Python
            const result = await processExcelFile(filePath, originalName);
            // Clean up the uploaded file 
            await fs.unlink(filePath);
            // Result of the file processing
            res.json({
                success: true,
                file_info: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            // Clean up file on error
            if (req.file) {
                try {
                    await fs.unlink(req.file.path);
                } catch (cleanupError) {
                    console.error('Error cleaning up file:', cleanupError);
                }
            }
            next(error);
        }
    }
);

// Analyze data in a sheet
router.get('/excel/:sessionId/analyze/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            
            const analysis = await analyzeSheetData(sessionId, sheetName);
            
            res.json({
                success: true,
                analysis: analysis,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Clean data in a sheet
router.post('/excel/:sessionId/clean/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    body('operations').isArray().withMessage('Operations must be an array'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { operations } = req.body;
            
            const result = await cleanSheetData(sessionId, sheetName, operations);
            
            res.json({
                success: true,
                result: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Create chart from sheet data
router.post('/excel/:sessionId/chart/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    body('chart_config').isObject().withMessage('Chart config must be an object'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { chart_config } = req.body;
            
            const chartData = await createChart(sessionId, sheetName, chart_config);
            
            res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
            res.setHeader('Content-Disposition', `attachment; filename="chart_${sheetName}.xlsx"`);
            res.send(chartData);
            
        } catch (error) {
            next(error);
        }
    }
);

// Export data in various formats
router.get('/excel/:sessionId/export/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    query('format').isIn(['csv', 'json', 'excel', 'parquet']).withMessage('Format must be csv, json, excel, or parquet'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { format } = req.query;
            
            const exportData = await exportSheetData(sessionId, sheetName, format);
            
            const contentType = {
                'csv': 'text/csv',
                'json': 'application/json',
                'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'parquet': 'application/octet-stream'
            };
            
            res.setHeader('Content-Type', contentType[format]);
            res.setHeader('Content-Disposition', `attachment; filename="${sheetName}.${format}"`);
            res.send(exportData);
            
        } catch (error) {
            next(error);
        }
    }
);

// Apply formulas to sheet data
router.post('/excel/:sessionId/formulas/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    body('formulas').isObject().withMessage('Formulas must be an object'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { formulas } = req.body;
            
            const result = await applyFormulas(sessionId, sheetName, formulas);
            
            res.json({
                success: true,
                result: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Get summary of all loaded data
router.get('/excel/:sessionId/summary',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId } = req.params;
            
            const summary = await getDataSummary(sessionId);
            
            res.json({
                success: true,
                summary: summary,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Get data preview for a sheet
router.get('/excel/:sessionId/preview/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    query('limit').optional().isInt({ min: 1, max: 1000 }).withMessage('Limit must be between 1 and 1000'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const limit = parseInt(req.query.limit) || 50;
            
            const previewData = await getDataPreview(sessionId, sheetName, limit);
            
            res.json({
                success: true,
                data: previewData,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Get column information for a sheet
router.get('/excel/:sessionId/columns/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            
            const columnInfo = await getColumnInfo(sessionId, sheetName);
            
            res.json({
                success: true,
                columns: columnInfo,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Validate data in a sheet
router.post('/excel/:sessionId/validate/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    body('rules').isArray().withMessage('Rules must be an array'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { rules } = req.body;
            
            const validationResults = await validateSheetData(sessionId, sheetName, rules);
            
            res.json({
                success: true,
                validation: validationResults,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Transform data in a sheet
router.post('/excel/:sessionId/transform/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    body('transformations').isArray().withMessage('Transformations must be an array'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { transformations } = req.body;
            
            const result = await transformSheetData(sessionId, sheetName, transformations);
            
            res.json({
                success: true,
                result: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Machine Learning Endpoints

// Predict values using ML
router.post('/excel/:sessionId/predict/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    body('target_column').isLength({ min: 1 }).withMessage('Target column is required'),
    body('feature_columns').isArray().withMessage('Feature columns must be an array'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { target_column, feature_columns, model_type = 'auto' } = req.body;
            
            const result = await predictValues(sessionId, sheetName, target_column, feature_columns, model_type);
            
            res.json({
                success: true,
                prediction: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Cluster data
router.post('/excel/:sessionId/cluster/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    body('feature_columns').isArray().withMessage('Feature columns must be an array'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { feature_columns, n_clusters = 3, algorithm = 'kmeans' } = req.body;
            
            const result = await clusterData(sessionId, sheetName, feature_columns, n_clusters, algorithm);
            
            res.json({
                success: true,
                clustering: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Detect anomalies
router.post('/excel/:sessionId/anomalies/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    body('feature_columns').isArray().withMessage('Feature columns must be an array'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { feature_columns, method = 'isolation_forest' } = req.body;
            
            const result = await detectAnomalies(sessionId, sheetName, feature_columns, method);
            
            res.json({
                success: true,
                anomalies: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Analyze correlations
router.post('/excel/:sessionId/correlation/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    body('columns').isArray().withMessage('Columns must be an array'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { columns } = req.body;
            
            const result = await analyzeCorrelations(sessionId, sheetName, columns);
            
            res.json({
                success: true,
                correlation: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Get ML recommendations
router.get('/excel/:sessionId/ml-recommendations/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            
            const result = await getMLRecommendations(sessionId, sheetName);
            
            res.json({
                success: true,
                recommendations: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Cloud Storage Endpoints

// Get cloud storage status
router.get('/cloud/status',
    async (req, res, next) => {
        try {
            const status = await getCloudStorageStatus();
            
            res.json({
                success: true,
                status: status,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Upload file to cloud storage
router.post('/cloud/upload',
    upload.single('file'),
    body('provider').isIn(['s3', 'google_drive', 'dropbox']).withMessage('Provider must be s3, google_drive, or dropbox'),
    body('cloud_path').optional().isLength({ min: 1 }).withMessage('Cloud path must be provided'),
    validateRequest,
    async (req, res, next) => {
        try {
            if (!req.file) {
                return res.status(400).json({
                    error: {
                        message: 'No file uploaded',
                        status: 400
                    }
                });
            }
            
            const { provider, cloud_path, metadata } = req.body;
            const file_path = req.file.path;
            
            const result = await uploadToCloud(file_path, provider, cloud_path, metadata);
            
            // Clean up uploaded file
            await fs.unlink(file_path);
            
            res.json({
                success: true,
                upload: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            // Clean up file on error
            if (req.file) {
                try {
                    await fs.unlink(req.file.path);
                } catch (cleanupError) {
                    console.error('Error cleaning up file:', cleanupError);
                }
            }
            next(error);
        }
    }
);

// Download file from cloud storage
router.get('/cloud/download/:provider',
    param('provider').isIn(['s3', 'google_drive', 'dropbox']).withMessage('Provider must be s3, google_drive, or dropbox'),
    query('cloud_path').isLength({ min: 1 }).withMessage('Cloud path is required'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { provider } = req.params;
            const { cloud_path } = req.query;
            
            const result = await downloadFromCloud(provider, cloud_path);
            
            if (result.success) {
                res.download(result.local_path, result.filename || 'downloaded_file');
            } else {
                res.status(400).json({
                    error: {
                        message: result.error,
                        status: 400
                    }
                });
            }
            
        } catch (error) {
            next(error);
        }
    }
);

// List files in cloud storage
router.get('/cloud/list/:provider',
    param('provider').isIn(['s3', 'google_drive', 'dropbox']).withMessage('Provider must be s3, google_drive, or dropbox'),
    query('prefix').optional().isLength({ min: 1 }).withMessage('Prefix must be provided'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { provider } = req.params;
            const { prefix } = req.query;
            
            const result = await listCloudFiles(provider, prefix);
            
            res.json({
                success: true,
                files: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Sync processed data to cloud
router.post('/excel/:sessionId/sync/:sheetName',
    param('sessionId').isLength({ min: 1 }).withMessage('Session ID is required'),
    param('sheetName').isLength({ min: 1 }).withMessage('Sheet name is required'),
    body('provider').isIn(['s3', 'google_drive', 'dropbox']).withMessage('Provider must be s3, google_drive, or dropbox'),
    body('format').isIn(['csv', 'json', 'excel']).withMessage('Format must be csv, json, or excel'),
    body('cloud_path').optional().isLength({ min: 1 }).withMessage('Cloud path must be provided'),
    validateRequest,
    async (req, res, next) => {
        try {
            const { sessionId, sheetName } = req.params;
            const { provider, format, cloud_path, metadata } = req.body;
            
            // First export the data
            const exportData = await exportSheetData(sessionId, sheetName, format);
            
            // Create temporary file
            const tempPath = path.join(__dirname, 'temp', `sync_${Date.now()}.${format}`);
            await fs.writeFile(tempPath, exportData);
            
            // Upload to cloud
            const result = await uploadToCloud(tempPath, provider, cloud_path, metadata);
            
            // Clean up temp file
            await fs.unlink(tempPath);
            
            res.json({
                success: true,
                sync: result,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            next(error);
        }
    }
);

// Batch process multiple files
router.post('/excel/batch',
    upload.array('files', 10), // Max 10 files
    async (req, res, next) => {
        try {
            if (!req.files || req.files.length === 0) {
                return res.status(400).json({
                    error: {
                        message: 'No files uploaded',
                        status: 400
                    }
                });
            }

            const results = await Promise.allSettled(
                req.files.map(file => processExcelFile(file.path, file.originalname))
            );
            
            // Clean up uploaded files
            await Promise.all(
                req.files.map(file => fs.unlink(file.path))
            );
            
            const processedResults = results.map((result, index) => ({
                file: req.files[index].originalname,
                success: result.status === 'fulfilled',
                data: result.status === 'fulfilled' ? result.value : null,
                error: result.status === 'rejected' ? result.reason.message : null
            }));
            
            res.json({
                success: true,
                results: processedResults,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            // Clean up files on error
            if (req.files) {
                await Promise.all(
                    req.files.map(file => fs.unlink(file.path).catch(console.error))
                );
            }
            next(error);
        }
    }
);

// Helper functions to interact with Python Excel processor
async function processExcelFile(filePath, originalName) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'load_file',
            filePath
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function analyzeSheetData(sessionId, sheetName) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'analyze_data',
            sessionId,
            sheetName
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function cleanSheetData(sessionId, sheetName, operations) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'clean_data',
            sessionId,
            sheetName,
            JSON.stringify(operations)
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function createChart(sessionId, sheetName, chartConfig) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'create_chart',
            sessionId,
            sheetName,
            JSON.stringify(chartConfig)
        ]);
        
        let output = Buffer.alloc(0);
        let error = '';
        
        python.stdout.on('data', (data) => {
            output = Buffer.concat([output, data]);
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                resolve(output);
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function exportSheetData(sessionId, sheetName, format) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'export_data',
            sessionId,
            sheetName,
            format
        ]);
        
        let output = Buffer.alloc(0);
        let error = '';
        
        python.stdout.on('data', (data) => {
            output = Buffer.concat([output, data]);
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                resolve(output);
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function applyFormulas(sessionId, sheetName, formulas) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'apply_formulas',
            sessionId,
            sheetName,
            JSON.stringify(formulas)
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function getDataSummary(sessionId) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'get_summary',
            sessionId
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function getDataPreview(sessionId, sheetName, limit) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'get_preview',
            sessionId,
            sheetName,
            limit.toString()
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function getColumnInfo(sessionId, sheetName) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'get_columns',
            sessionId,
            sheetName
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function validateSheetData(sessionId, sheetName, rules) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'validate_data',
            sessionId,
            sheetName,
            JSON.stringify(rules)
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function transformSheetData(sessionId, sheetName, transformations) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'excel_processor.py'),
            'transform_data',
            sessionId,
            sheetName,
            JSON.stringify(transformations)
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

// Machine Learning helper functions
async function predictValues(sessionId, sheetName, targetColumn, featureColumns, modelType) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'ml_processor.py'),
            'predict',
            sheetName,
            targetColumn,
            JSON.stringify(featureColumns),
            modelType
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function clusterData(sessionId, sheetName, featureColumns, nClusters, algorithm) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'ml_processor.py'),
            'cluster',
            sheetName,
            JSON.stringify(featureColumns),
            nClusters.toString()
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function detectAnomalies(sessionId, sheetName, featureColumns, method) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'ml_processor.py'),
            'anomalies',
            sheetName,
            JSON.stringify(featureColumns)
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function analyzeCorrelations(sessionId, sheetName, columns) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [
            path.join(__dirname, 'ml_processor.py'),
            'correlation',
            sheetName,
            JSON.stringify(columns)
        ]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Python process failed: ${error}`));
            }
        });
    });
}

async function getMLRecommendations(sessionId, sheetName) {
    return new Promise((resolve, reject) => {
        // First get data summary
        getDataSummary(sessionId).then(summary => {
            const python = spawn('python3', [
                path.join(__dirname, 'ml_processor.py'),
                'recommendations',
                sheetName,
                JSON.stringify(summary)
            ]);
            
            let output = '';
            let error = '';
            
            python.stdout.on('data', (data) => {
                output += data.toString();
            });
            
            python.stderr.on('data', (data) => {
                error += data.toString();
            });
            
            python.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(output);
                        resolve(result);
                    } catch (parseError) {
                        reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                    }
                } else {
                    reject(new Error(`Python process failed: ${error}`));
                }
            });
        }).catch(reject);
    });
}

module.exports = router;
