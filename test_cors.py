#!/usr/bin/env python3
import requests
import json
import sys

def test_cors_headers(api_url, origin):
    """Test CORS headers for API endpoints"""
    print(f"Testing CORS headers for API: {api_url}")
    print(f"Using origin: {origin}")
    
    # Define endpoints to test
    endpoints = [
        {'name': 'Upload', 'path': 'upload', 'method': 'OPTIONS'},
        {'name': 'Analyze', 'path': 'analyze', 'method': 'OPTIONS'},
        {'name': 'Results', 'path': 'results/test', 'method': 'OPTIONS'}
    ]
    
    all_passed = True
    
    for endpoint in endpoints:
        url = f"{api_url.rstrip('/')}/{endpoint['path']}"
        print(f"\nTesting {endpoint['name']} endpoint: {url}")
        
        headers = {
            'Origin': origin,
            'Access-Control-Request-Method': 'POST' if endpoint['method'] == 'OPTIONS' else 'GET',
            'Access-Control-Request-Headers': 'Content-Type,X-Requested-With'
        }
        
        try:
            response = requests.options(url, headers=headers, timeout=10)
            print(f"Status code: {response.status_code}")
            
            # Print all headers for debugging
            print("Response headers:")
            for header, value in response.headers.items():
                print(f"  {header}: {value}")
            
            # Check for required CORS headers
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers'
            ]
            
            missing_headers = []
            for header in cors_headers:
                if header not in response.headers:
                    missing_headers.append(header)
            
            if missing_headers:
                print(f"❌ Missing CORS headers: {', '.join(missing_headers)}")
                all_passed = False
            else:
                # Check if origin is allowed
                if response.headers.get('Access-Control-Allow-Origin') != origin and response.headers.get('Access-Control-Allow-Origin') != '*':
                    print(f"❌ Origin {origin} not allowed. Allowed origin: {response.headers.get('Access-Control-Allow-Origin')}")
                    all_passed = False
                else:
                    print(f"✅ Origin {origin} is allowed")
                
                # Check if methods are allowed
                allowed_methods = response.headers.get('Access-Control-Allow-Methods', '')
                if endpoint['method'] not in allowed_methods and 'OPTIONS' not in allowed_methods:
                    print(f"❌ Method {endpoint['method']} not allowed. Allowed methods: {allowed_methods}")
                    all_passed = False
                else:
                    print(f"✅ Method {endpoint['method']} is allowed")
                
                # Check if headers are allowed
                allowed_headers = response.headers.get('Access-Control-Allow-Headers', '')
                if 'Content-Type' not in allowed_headers and '*' not in allowed_headers:
                    print(f"❌ Header Content-Type not allowed. Allowed headers: {allowed_headers}")
                    all_passed = False
                else:
                    print(f"✅ Header Content-Type is allowed")
                
                if all([
                    response.headers.get('Access-Control-Allow-Origin') in [origin, '*'],
                    endpoint['method'] in allowed_methods or 'OPTIONS' in allowed_methods,
                    'Content-Type' in allowed_headers or '*' in allowed_headers
                ]):
                    print(f"✅ {endpoint['name']} endpoint passed CORS checks")
                else:
                    print(f"❌ {endpoint['name']} endpoint failed CORS checks")
                    all_passed = False
        
        except Exception as e:
            print(f"❌ Error testing {endpoint['name']} endpoint: {str(e)}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_cors.py <api_url> <origin>")
        print("Example: python test_cors.py https://15zjlknmve.execute-api.us-east-1.amazonaws.com/Prod/ https://jacobeee.github.io")
        sys.exit(1)
    
    api_url = sys.argv[1]
    origin = sys.argv[2]
    
    success = test_cors_headers(api_url, origin)
    
    if success:
        print("\n✅ All CORS checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Some CORS checks failed!")
        sys.exit(1)