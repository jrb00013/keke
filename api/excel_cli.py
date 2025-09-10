#!/usr/bin/env python3
"""
Command-line interface for Excel processor
This script allows the Excel processor to be called from Node.js
"""

import sys
import json
import argparse
from excel_processor import ExcelProcessor

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command specified"}), file=sys.stderr)
        sys.exit(1)
    
    command = sys.argv[1]
    processor = ExcelProcessor()
    
    try:
        if command == 'load_file':
            if len(sys.argv) < 3:
                print(json.dumps({"error": "File path required"}), file=sys.stderr)
                sys.exit(1)
            
            file_path = sys.argv[2]
            result = processor.load_file(file_path)
            print(json.dumps(result))
            
        elif command == 'analyze_data':
            if len(sys.argv) < 4:
                print(json.dumps({"error": "Session ID and sheet name required"}), file=sys.stderr)
                sys.exit(1)
            
            session_id = sys.argv[2]
            sheet_name = sys.argv[3]
            
            # For CLI, we'll simulate session management
            # In production, you'd have proper session storage
            result = processor.analyze_data(sheet_name)
            print(json.dumps(result))
            
        elif command == 'clean_data':
            if len(sys.argv) < 5:
                print(json.dumps({"error": "Session ID, sheet name, and operations required"}), file=sys.stderr)
                sys.exit(1)
            
            session_id = sys.argv[2]
            sheet_name = sys.argv[3]
            operations = json.loads(sys.argv[4])
            
            result = processor.clean_data(sheet_name, operations)
            print(json.dumps(result))
            
        elif command == 'create_chart':
            if len(sys.argv) < 5:
                print(json.dumps({"error": "Session ID, sheet name, and chart config required"}), file=sys.stderr)
                sys.exit(1)
            
            session_id = sys.argv[2]
            sheet_name = sys.argv[3]
            chart_config = json.loads(sys.argv[4])
            
            chart_data = processor.create_chart(sheet_name, chart_config)
            # Output binary data to stdout
            sys.stdout.buffer.write(chart_data)
            
        elif command == 'export_data':
            if len(sys.argv) < 5:
                print(json.dumps({"error": "Session ID, sheet name, and format required"}), file=sys.stderr)
                sys.exit(1)
            
            session_id = sys.argv[2]
            sheet_name = sys.argv[3]
            format_type = sys.argv[4]
            
            export_data = processor.export_data(sheet_name, format_type)
            sys.stdout.buffer.write(export_data)
            
        elif command == 'apply_formulas':
            if len(sys.argv) < 5:
                print(json.dumps({"error": "Session ID, sheet name, and formulas required"}), file=sys.stderr)
                sys.exit(1)
            
            session_id = sys.argv[2]
            sheet_name = sys.argv[3]
            formulas = json.loads(sys.argv[4])
            
            result = processor.apply_formulas(sheet_name, formulas)
            print(json.dumps(result))
            
        elif command == 'get_summary':
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Session ID required"}), file=sys.stderr)
                sys.exit(1)
            
            session_id = sys.argv[2]
            result = processor.get_summary()
            print(json.dumps(result))
            
        else:
            print(json.dumps({"error": f"Unknown command: {command}"}), file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
