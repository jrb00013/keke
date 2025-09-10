# Keke - Advanced AI-Powered Excel Datasheet Tool

Keke is a cutting-edge, AI-powered web-based tool designed to revolutionize Excel functionality with advanced data processing, machine learning, and intelligent analysis capabilities. It provides a modern interface for working with Excel files, CSV data, and JSON files, enhanced with artificial intelligence for smarter data insights.

## Key Features

### AI-Powered Capabilities
- **Natural Language Queries**: Ask questions about your data in plain English
- **AI Data Analysis**: Intelligent insights and pattern recognition
- **Smart Data Cleaning**: AI-driven recommendations for data quality improvements
- **Automated Visualization**: AI suggests the best charts and graphs for your data
- **Predictive Analytics**: Machine learning models for forecasting and classification

### Advanced Data Processing
- **Multi-format Support**: Excel (.xlsx, .xls), CSV, JSON, Parquet
- **Real-time Collaboration**: Multiple users can work on the same dataset simultaneously
- **Cloud Integration**: Seamless integration with AWS S3, Google Drive, Dropbox
- **Batch Processing**: Handle multiple files simultaneously
- **Advanced Formulas**: Excel-like formula engine with AI enhancements

### Enterprise Features
- **Scalable Architecture**: Kubernetes-ready with auto-scaling
- **Comprehensive Monitoring**: Prometheus and Grafana integration
- **Security**: JWT authentication, rate limiting, data encryption
- **CI/CD Pipeline**: Automated testing and deployment
- **Multi-environment Support**: Development, staging, and production deployments

## Core Features

### Data Analysis
- Comprehensive data analysis with statistics and quality metrics
- Pattern detection including outliers and correlations
- Data type analysis and recommendations
- Memory usage optimization insights

### Data Cleaning
- Remove duplicates and empty rows
- Handle missing values with multiple strategies
- Data type conversion and validation
- Column renaming and restructuring

### Chart Generation
- Create professional charts (Bar, Line, Pie)
- Customizable chart styling and formatting
- Export charts as Excel files
- Multiple chart types support

### Export Options
- Export data in multiple formats: CSV, JSON, Excel, Parquet
- Batch processing for multiple files
- Custom export configurations
- High-performance data serialization

### Formula Engine
- Excel-like formula evaluation
- Support for common functions (SUM, AVERAGE, COUNT)
- Arithmetic operations between cells
- Custom formula application

### Batch Processing
- Process multiple files simultaneously
- Parallel processing for efficiency
- Progress tracking and error handling
- Bulk operations support

## Installation & Setup

### Prerequisites
- **Node.js** 18+ 
- **Python** 3.9+
- **Docker** (for containerized deployment)
- **Kubernetes** (for production deployment)
- **npm** or **yarn**

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/keke-team/keke-excel-tool.git
   cd keke-excel-tool
   ```

2. **Install dependencies**
   ```bash
   # Install all dependencies automatically
   python3 run.py install
   
   # Or install manually
   npm install
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Copy environment template
   cp env.example .env
   
   # Edit .env with your configuration
   nano .env
   ```

4. **Start Keke**
   ```bash
   # Start with all services (databases, monitoring)
   python3 run.py start --mode full
   
   # Or start application only
   python3 run.py start --mode app-only
   ```

### Docker Deployment

```bash
# Build Docker image
docker build -t keke-excel-tool .

# Run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
```

### Kubernetes Deployment

```bash
# Deploy to development
./scripts/deploy.sh deploy development

# Deploy to production
./scripts/deploy.sh deploy production

# Check deployment status
./scripts/deploy.sh status

# Rollback if needed
./scripts/deploy.sh rollback
```

### Manual Setup

1. **Install Node.js dependencies**
   ```bash
   npm install
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories**
   ```bash
   mkdir -p uploads logs data temp models cache
   ```

4. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export ANTHROPIC_API_KEY="your_anthropic_api_key"
   export DATABASE_URL="postgresql://user:pass@localhost:5432/keke"
   ```

## Usage

### Starting the Server

```bash
# Using the comprehensive runner (recommended)
python3 run.py start

# Development mode with auto-reload
npm run dev

# Production mode
npm start

# With specific configuration
python3 run.py start --mode with-db --verbose
```

The server will start on `http://localhost:3000`

### AI Assistant Usage

```bash
# Ask questions about your data
curl -X POST http://localhost:3000/api/ai/query/session123/Sheet1 \
  -H "Content-Type: application/json" \
  -d '{"query": "What insights can you provide about this sales data?"}'

# Get AI-powered cleaning suggestions
curl http://localhost:3000/api/ai/cleaning-suggestions/session123/Sheet1

# Get visualization recommendations
curl http://localhost:3000/api/ai/visualization-suggestions/session123/Sheet1
```

### Web Interface

1. Open your browser and navigate to `http://localhost:3000`
2. Upload your Excel, CSV, or JSON file using drag-and-drop or file picker
3. Use the available tools to analyze, clean, and process your data
4. Export results in your preferred format

### API Endpoints

#### File Upload
```http
POST /api/excel/upload
Content-Type: multipart/form-data

file: [your file]
```

#### Data Analysis
```http
GET /api/excel/{sessionId}/analyze/{sheetName}
```

#### Data Cleaning
```http
POST /api/excel/{sessionId}/clean/{sheetName}
Content-Type: application/json

{
  "operations": [
    {
      "type": "remove_duplicates"
    },
    {
      "type": "remove_nulls",
      "params": {
        "strategy": "drop_rows"
      }
    }
  ]
}
```

#### Chart Creation
```http
POST /api/excel/{sessionId}/chart/{sheetName}
Content-Type: application/json

{
  "chart_config": {
    "type": "bar",
    "title": "Sales Data",
    "x_column": "Month",
    "y_columns": ["Sales", "Profit"]
  }
}
```

#### Data Export
```http
GET /api/excel/{sessionId}/export/{sheetName}?format=csv
```

#### Formula Application
```http
POST /api/excel/{sessionId}/formulas/{sheetName}
Content-Type: application/json

{
  "formulas": {
    "Total": "=SUM(A:A)",
    "Average": "=AVERAGE(B:B)"
  }
}
```

#### Machine Learning Predictions
```http
POST /api/excel/{sessionId}/predict/{sheetName}
Content-Type: application/json

{
  "target_column": "Sales",
  "feature_columns": ["Price", "Marketing", "Season"],
  "model_type": "auto"
}
```

#### AI Query Processing
```http
POST /api/ai/query/{sessionId}/{sheetName}
Content-Type: application/json

{
  "query": "What are the main trends in this data?",
  "context": {}
}
```

## Supported File Formats

- **Excel**: .xlsx, .xls
- **CSV**: .csv (with automatic delimiter detection)
- **JSON**: .json (arrays of objects or single objects)
- **Parquet**: .parquet (for high-performance data processing)

## Data Cleaning Operations

### Remove Duplicates
```json
{
  "type": "remove_duplicates"
}
```

### Handle Missing Values
```json
{
  "type": "remove_nulls",
  "params": {
    "strategy": "drop_rows" | "drop_columns" | "fill",
    "threshold": 0.5,
    "method": "forward" | "backward" | "mean" | "median"
  }
}
```

### Convert Data Types
```json
{
  "type": "convert_types",
  "params": {
    "conversions": {
      "column_name": "numeric" | "datetime" | "string"
    }
  }
}
```

### Rename Columns
```json
{
  "type": "rename_columns",
  "params": {
    "mapping": {
      "old_name": "new_name"
    }
  }
}
```

### Filter Rows
```json
{
  "type": "filter_rows",
  "params": {
    "condition": "column_name > 100"
  }
}
```

## Chart Configuration

### Chart Types
- `bar`: Bar chart
- `line`: Line chart  
- `pie`: Pie chart

### Configuration Options
```json
{
  "type": "bar",
  "title": "Chart Title",
  "x_column": "Category",
  "y_columns": ["Value1", "Value2"],
  "style": 10
}
```

## Formula Support

### Supported Functions
- `SUM(range)`: Sum of values in range
- `AVERAGE(range)`: Average of values in range
- `COUNT(range)`: Count of non-empty cells in range

### Cell References
- `A1`, `B2`: Individual cell references
- `A:A`: Entire column reference
- `1:1`: Entire row reference

### Arithmetic Operations
- `A1+B1`: Addition
- `A1-B1`: Subtraction
- `A1*B1`: Multiplication
- `A1/B1`: Division

## Machine Learning Features

### Predictive Models
- **Auto Model Selection**: Automatically chooses the best model for your data
- **Supported Models**: Linear Regression, Random Forest, XGBoost, LightGBM
- **Cross-validation**: Comprehensive model evaluation
- **Feature Importance**: Analysis of which features matter most

### Clustering
- **K-Means Clustering**: Group similar data points
- **Hierarchical Clustering**: Tree-based clustering
- **DBSCAN**: Density-based clustering
- **Auto Parameter Tuning**: Automatic selection of optimal parameters

### Data Validation
- **Statistical Validation**: Comprehensive data quality checks
- **Custom Rules**: Define your own validation rules
- **Anomaly Detection**: Identify outliers and unusual patterns
- **Data Profiling**: Detailed analysis of data characteristics

## Performance Considerations

- **File Size**: Recommended maximum 50MB per file
- **Memory Usage**: Large files are processed in chunks
- **Concurrent Requests**: Rate limited to 100 requests per 15 minutes
- **Batch Processing**: Up to 10 files simultaneously

## Error Handling

The API provides comprehensive error handling with:
- Detailed error messages
- HTTP status codes
- Validation error details
- File processing error recovery

## Development

### Running Tests
```bash
# Run all tests
python3 run.py test

# Run Node.js tests
npm test

# Run Python tests
pytest
```

### Linting
```bash
npm run lint
```

### Code Formatting
```bash
npm run format
```

### Python Testing
```bash
pytest
```

## Project Structure

```
keke/
├── api/                    # API endpoints and server code
│   ├── server.js          # Main Express server
│   ├── api_routes.js      # API route definitions
│   ├── excel_processor.py # Core Excel processing logic
│   ├── assistant.py       # AI assistant functionality
│   ├── ml_processor.py    # Machine learning processing
│   ├── security.py        # Security utilities
│   ├── cloud_services.py  # Cloud storage integration
│   └── public/            # Static web interface
├── k8s/                   # Kubernetes deployment manifests
├── monitoring/            # Monitoring configuration
├── scripts/               # Deployment and utility scripts
├── tests/                 # Test suites
├── run.py                 # Main startup script
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile            # Docker image definition
├── requirements.txt      # Python dependencies
├── package.json         # Node.js dependencies
└── env.example         # Environment configuration template
```

## Configuration

### Environment Variables

#### Application Configuration
- `KEKE_ENV`: Environment (development, staging, production)
- `HOST`: Server host (default: localhost)
- `PORT`: Server port (default: 3000)
- `DEBUG`: Enable debug mode

#### AI Configuration
- `AI_ENABLED`: Enable AI features
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic Claude API key

#### Database Configuration
- `DATABASE_URL`: Primary database connection string
- `REDIS_URL`: Redis connection string
- `MONGODB_URL`: MongoDB connection string
- `POSTGRES_URL`: PostgreSQL connection string

#### Cloud Storage Configuration
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region
- `AZURE_STORAGE_CONNECTION_STRING`: Azure storage connection
- `GOOGLE_APPLICATION_CREDENTIALS`: Google Cloud credentials

#### Security Configuration
- `JWT_SECRET`: JWT signing secret
- `ENCRYPTION_KEY`: Data encryption key
- `ALLOWED_ORIGINS`: CORS allowed origins
- `RATE_LIMIT`: API rate limit
- `RATE_WINDOW`: Rate limit window

#### Performance Configuration
- `MAX_FILE_SIZE`: Maximum file size in bytes
- `WORKER_PROCESSES`: Number of worker processes
- `MAX_MEMORY_USAGE`: Memory limit
- `CACHE_TTL`: Cache time-to-live

### Feature Flags
- `FEATURE_AI_ASSISTANT`: Enable AI assistant
- `FEATURE_MACHINE_LEARNING`: Enable ML features
- `FEATURE_CLOUD_STORAGE`: Enable cloud storage
- `FEATURE_COLLABORATION`: Enable real-time collaboration
- `FEATURE_ADVANCED_ANALYTICS`: Enable advanced analytics

## Security

### Authentication & Authorization
- JWT-based authentication
- Role-based access control
- API rate limiting
- CORS configuration

### Data Protection
- Encryption at rest and in transit
- Secure secret management
- Input validation and sanitization
- SQL injection prevention

### Infrastructure Security
- Non-root containers
- Network policies
- Pod security policies
- Regular security scanning

## Monitoring & Observability

### Prometheus Metrics
- Application performance metrics
- Custom business metrics
- Kubernetes cluster metrics
- Database performance metrics

### Grafana Dashboards
- Application health monitoring
- Performance analytics
- Error tracking and alerting
- Resource utilization

### Logging
- Structured JSON logging
- Centralized log aggregation
- Error tracking and debugging
- Performance monitoring

## Scalability

### Horizontal Scaling
- Kubernetes HPA for automatic scaling
- Load balancer configuration
- Multi-replica deployments
- Database connection pooling

### Performance Optimization
- Caching strategies (Redis)
- CDN integration
- Resource optimization
- Database indexing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API examples

## Roadmap

### Upcoming Features
- Advanced data visualization
- Enhanced machine learning integration
- Real-time collaboration improvements
- Advanced cloud storage integration
- Extended formula functions
- Data validation rules
- Automated reporting
- API rate limiting improvements

### Version History
- **v1.0.0**: Initial release with core Excel processing features
- **v1.1.0**: Added chart generation and advanced data cleaning
- **v1.2.0**: Enhanced formula engine and batch processing
- **v1.3.0**: AI assistant and machine learning integration

---

**Keke** - Making Excel data processing simple, powerful, and efficient.