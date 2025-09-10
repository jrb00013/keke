# Keke - Advanced AI-Powered Excel Datasheet Tool

Keke is a cutting-edge, AI-powered web-based tool designed to revolutionize Excel functionality with advanced data processing, machine learning, and intelligent analysis capabilities. It provides a modern interface for working with Excel files, CSV data, and JSON files, enhanced with artificial intelligence for smarter data insights.

## 🚀 Key Features

### 🤖 AI-Powered Capabilities
- **Natural Language Queries**: Ask questions about your data in plain English
- **AI Data Analysis**: Intelligent insights and pattern recognition
- **Smart Data Cleaning**: AI-driven recommendations for data quality improvements
- **Automated Visualization**: AI suggests the best charts and graphs for your data
- **Predictive Analytics**: Machine learning models for forecasting and classification

### 📊 Advanced Data Processing
- **Multi-format Support**: Excel (.xlsx, .xls), CSV, JSON, Parquet
- **Real-time Collaboration**: Multiple users can work on the same dataset simultaneously
- **Cloud Integration**: Seamless integration with AWS S3, Google Drive, Dropbox
- **Batch Processing**: Handle multiple files simultaneously
- **Advanced Formulas**: Excel-like formula engine with AI enhancements

### 🔧 Enterprise Features
- **Scalable Architecture**: Kubernetes-ready with auto-scaling
- **Comprehensive Monitoring**: Prometheus and Grafana integration
- **Security**: JWT authentication, rate limiting, data encryption
- **CI/CD Pipeline**: Automated testing and deployment
- **Multi-environment Support**: Development, staging, and production deployments

## Features

### 📊 Data Analysis
- Comprehensive data analysis with statistics and quality metrics
- Pattern detection including outliers and correlations
- Data type analysis and recommendations
- Memory usage optimization insights

### 🧹 Data Cleaning
- Remove duplicates and empty rows
- Handle missing values with multiple strategies
- Data type conversion and validation
- Column renaming and restructuring

### 📈 Chart Generation
- Create professional charts (Bar, Line, Pie)
- Customizable chart styling and formatting
- Export charts as Excel files
- Multiple chart types support

### 📤 Export Options
- Export data in multiple formats: CSV, JSON, Excel, Parquet
- Batch processing for multiple files
- Custom export configurations
- High-performance data serialization

### ⚡ Formula Engine
- Excel-like formula evaluation
- Support for common functions (SUM, AVERAGE, COUNT)
- Arithmetic operations between cells
- Custom formula application

### 🔄 Batch Processing
- Process multiple files simultaneously
- Parallel processing for efficiency
- Progress tracking and error handling
- Bulk operations support

## 🛠️ Installation & Setup

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

### 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t keke-excel-tool .

# Run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
```

### ☸️ Kubernetes Deployment

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

### 🔧 Manual Setup

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

## 🚀 Usage

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

## Supported File Formats

- **Excel**: .xlsx, .xls
- **CSV**: .csv (with automatic delimiter detection)
- **JSON**: .json (arrays of objects or single objects)

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
npm test
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
- [ ] Advanced data visualization
- [ ] Machine learning integration
- [ ] Real-time collaboration
- [ ] Cloud storage integration
- [ ] Advanced formula functions
- [ ] Data validation rules
- [ ] Automated reporting
- [ ] API rate limiting improvements

### Version History
- **v1.0.0**: Initial release with core Excel processing features
- **v1.1.0**: Added chart generation and advanced data cleaning
- **v1.2.0**: Enhanced formula engine and batch processing

---

**Keke** - Making Excel data processing simple, powerful, and efficient.