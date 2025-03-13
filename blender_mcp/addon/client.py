import socket
import json
import struct

def send_command(host='localhost', port=9876, command=None, timeout=30):
    """Send a command to the Blender MCP server and return the response."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)  # Increased timeout to 30 seconds
        s.connect((host, port))
        # Prepare message with length prefix
        message = json.dumps(command).encode('utf-8')
        length_prefix = struct.pack('>I', len(message))
        s.sendall(length_prefix + message)
        # Receive response
        length_data = s.recv(4)
        if not length_data:
            raise ConnectionError("Connection closed before receiving data")
        length = struct.unpack('>I', length_data)[0]
        response_data = s.recv(length)
        return json.loads(response_data.decode('utf-8'))

# Example usage
command = {"type": "get_scene_info", "params": {}}
try:
    response = send_command(command=command, timeout=30)
    print("Response:", response)
except socket.timeout:
    print("Error: Request timed out")
except Exception as e:
    print(f"Error: {str(e)}")