from mcp.server.fastmcp import FastMCP, Context
import socket
import json
import logging
import time
import os
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import base64

# Configure logging
LOG_DIR = "/tmp"  # Adjust this path as needed
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "blender_mcp_server.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BlenderMCPServer")

@dataclass
class BlenderConnection:
    host: str = "localhost"
    port: int = 9876
    sock: Optional[socket.socket] = None
    timeout: float = 60.0  # Increased from 15.0 for longer operations
    max_reconnect_attempts: int = 3
    base_reconnect_delay: float = 1.0

    def connect(self) -> bool:
        """Establish a connection to the Blender addon."""
        if self.sock:
            return True
        for attempt in range(self.max_reconnect_attempts):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                logger.info(f"Connected to Blender at {self.host}:{self.port}")
                return True
            except Exception as e:
                delay = self.base_reconnect_delay * (2 ** attempt)
                logger.error(f"Failed to connect (attempt {attempt + 1}/{self.max_reconnect_attempts}): {str(e)}")
                if attempt < self.max_reconnect_attempts - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("Max reconnect attempts reached")
                    self.sock = None
                    return False
        return False

    def disconnect(self):
        """Close the connection to Blender."""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Blender: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self) -> bytes:
        """Receive a complete JSON response from Blender."""
        chunks = []
        start_time = time.time()
        while True:
            try:
                chunk = self.sock.recv(8192)
                if not chunk:
                    if not chunks:
                        raise Exception("Connection closed before receiving data")
                    break
                chunks.append(chunk)
                try:
                    data = b''.join(chunks)
                    json.loads(data.decode('utf-8'))
                    logger.info(f"Received complete response ({len(data)} bytes)")
                    return data
                except json.JSONDecodeError:
                    continue
            except socket.timeout:
                elapsed = time.time() - start_time
                logger.warning(f"Socket timeout after {elapsed:.2f} seconds")
                if chunks:
                    data = b''.join(chunks)
                    try:
                        json.loads(data.decode('utf-8'))
                        logger.info(f"Recovered partial response ({len(data)} bytes)")
                        return data
                    except json.JSONDecodeError:
                        raise Exception("Incomplete JSON response received")
                raise Exception("No data received within timeout")
            except Exception as e:
                logger.error(f"Error during receive: {str(e)}")
                raise

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Blender and return the response."""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Blender")
        command = {"type": command_type, "params": params or {}}
        try:
            logger.info(f"Sending command: {command_type} with params: {params}")
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            response_data = self.receive_full_response()
            response = json.loads(response_data.decode('utf-8'))
            if response.get("status") == "error":
                logger.error(f"Blender error: {response.get('message')}")
                raise Exception(response.get("message", "Unknown error"))
            return response.get("result", {})
        except Exception as e:
            logger.error(f"Error communicating with Blender: {str(e)}", exc_info=True)
            self.disconnect()
            raise Exception(f"Connection to Blender lost: {str(e)}")

# Global connection management
_blender_connection = None

def get_blender_connection(host: str = "localhost", port: int = 9876) -> BlenderConnection:
    """Get or create a connection to Blender."""
    global _blender_connection
    if _blender_connection is not None:
        try:
            _blender_connection.sock.sendall(b'')
            return _blender_connection
        except Exception:
            logger.warning("Existing connection is no longer valid")
            _blender_connection.disconnect()
            _blender_connection = None
    _blender_connection = BlenderConnection(host=host, port=port)
    if not _blender_connection.connect():
        logger.error("Failed to connect to Blender")
        _blender_connection = None
        raise ConnectionError("Could not connect to Blender. Ensure the addon is running.")
    return _blender_connection

@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Manage the server's lifecycle."""
    logger.info("BlenderMCP server starting up")
    try:
        get_blender_connection()  # Verify connection on startup
        logger.info("Successfully connected to Blender")
        yield {}
    except Exception as e:
        logger.warning(f"Could not connect to Blender on startup: {str(e)}")
        logger.warning("Ensure the Blender addon is running before using tools")
        yield {}
    finally:
        global _blender_connection
        if _blender_connection:
            logger.info("Disconnecting from Blender on shutdown")
            _blender_connection.disconnect()
            _blender_connection = None
        logger.info("BlenderMCP server shut down")

# Create the MCP server
mcp = FastMCP(
    "BlenderMCP",
    description="Blender integration for dynamic scene manipulation via MCP",
    lifespan=server_lifespan
)

### Core Tools ###

@mcp.tool()
def get_scene_info(
    ctx: Context,
    filters: Dict[str, Any] = None,
    properties: List[str] = None,
    sub_object_data: Dict[str, Any] = None,
    limit: int = None,
    offset: int = 0,
    timeout: float = 5.0
) -> Dict[str, Any]:
    """
    Retrieve detailed information about the current Blender scene with advanced filtering and efficiency options.

    Parameters:
        filters (dict, optional): Filters to narrow down objects.
            - "type": Object type (e.g., "MESH", "LIGHT", "CAMERA").
            - "name_contains": Substring in object name.
            - "spatial_bounds": Dict with "min" and "max" coordinates (e.g., {"min": [-1, -1, -1], "max": [1, 1, 1]}).
        properties (list, optional): Properties to include (e.g., ["name", "location", "vertices"]).
            - Options: "name", "type", "location", "rotation", "scale", "vertex_count", "face_count", "vertices", "modifiers".
        sub_object_data (dict, optional): Options for sub-object data like vertices.
            - "vertices": {"sample_rate": 0.1, "max_count": 1000} (sample 10% or cap at 1000 vertices).
        limit (int, optional): Max number of objects to return (pagination).
        offset (int, optional): Starting index for pagination (default: 0).
        timeout (float, optional): Max time in seconds (default: 5.0).

    Returns:
        dict: Scene data or an error with a suggestion if timed out.

    Examples:
        - Get all meshes: {"filters": {"type": "MESH"}}
        - Get vertices for a cube: {"filters": {"name_contains": "Cube"}, "properties": ["vertices"]}
        - Limit to 10 objects with timeout: {"limit": 10, "timeout": 3.0}
    """
    try:
        blender = get_blender_connection()
        params = {
            "filters": filters or {},
            "properties": properties or ["name", "type", "location"],
            "sub_object_data": sub_object_data or {},
            "limit": limit,
            "offset": offset,
            "timeout": timeout
        }
        result = blender.send_command("get_scene_info", params)
        return result
    except Exception as e:
        logger.error(f"Error getting scene info: {str(e)}")
        return {
            "error": str(e),
            "suggestion": "Try applying more specific filters or increasing the timeout."
        }

@mcp.tool()
def run_script(ctx: Context, script: str) -> str:
    """
    Execute a Python script in Blender to manipulate the scene.

    Parameters:
        script: str
            A string containing the Python script to execute in Blender.
            The script should use Blender's Python API (bpy) to perform operations.
            Example:
                '''
                import bpy
                bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
                cube = bpy.context.object
                cube.name = "MyCube"
                '''

    Returns:
        A string confirming the script execution or an error message if it fails.
    """
    try:
        blender = get_blender_connection()
        # Encode script in base64 to prevent transmission issues
        script_encoded = base64.b64encode(script.encode('utf-8')).decode('ascii')
        result = blender.send_command("run_script", {"script": script_encoded})
        return f"Script executed successfully: {result.get('message', 'No message returned')}"
    except Exception as e:
        logger.error(f"Error running script: {str(e)}")
        return f"Error: {str(e)}"

def main():
    """Run the FastMCP server."""
    mcp.run()

if __name__ == "__main__":
    main()