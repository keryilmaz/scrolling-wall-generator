#!/usr/bin/env python
"""Robust server runner with auto-restart and error handling"""

import subprocess
import time
import sys
import os
import signal

def run_server():
    """Run the scrolling wall server with error handling"""
    print("Starting Scrolling Wall Generator Server...")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    while True:
        try:
            # Run the server
            process = subprocess.Popen(
                [sys.executable, "scrolling_wall_gradio.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output
            for line in process.stdout:
                print(line, end='')
                
                # Check for successful startup
                if "Running on" in line:
                    print("\n✅ Server is running successfully!")
                    print("Access the app at: http://localhost:7777")
                    print("-" * 50)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                print(f"\n⚠️ Server stopped with code {return_code}")
                print("Restarting in 5 seconds...")
                time.sleep(5)
            else:
                print("\n✅ Server stopped normally")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down server...")
            if 'process' in locals():
                process.terminate()
                process.wait()
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Activate virtual environment if needed
    venv_python = os.path.join("venv", "bin", "python")
    if os.path.exists(venv_python):
        sys.executable = venv_python
    
    run_server()