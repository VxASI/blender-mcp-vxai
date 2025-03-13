from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import logging
import time
import os
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

# Configure logging with file output
LOG_DIR = "/tmp"  # Adjust as needed for your system
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
    timeout: float = 15.0  # Consistent timeout in seconds
    max_reconnect_attempts: int = 3
    base_reconnect_delay: float = 1.0  # Seconds

    def connect(self) -> bool:
        """Connect to the Blender addon socket server with exponential backoff"""
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
                delay = self.base_reconnect_delay * (2 ** attempt)  # Exponential backoff
                logger.error(f"Failed to connect to Blender (attempt {attempt + 1}/{self.max_reconnect_attempts}): {str(e)}")
                if attempt < self.max_reconnect_attempts - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("Max reconnect attempts reached")
                    self.sock = None
                    return False
        return False

    def disconnect(self):
        """Disconnect from the Blender addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Blender: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self) -> bytes:
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        try:
            while True:
                chunk = self.sock.recv(8192)
                if not chunk:
                    if not chunks:
                        raise Exception("Connection closed before receiving any data")
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
            logger.warning("Socket timeout during receive")
            if chunks:
                data = b''.join(chunks)
                try:
                    json.loads(data.decode('utf-8'))
                    return data
                except json.JSONDecodeError:
                    raise Exception("Incomplete JSON response received")
            raise Exception("No data received within timeout")
        except Exception as e:
            logger.error(f"Error during receive: {str(e)}")
            raise

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a single command to Blender and return the response"""
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

    def send_batch(self, commands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Send a batch of commands to Blender"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Blender")

        batch_command = {"type": "batch", "params": {"commands": commands}}
        try:
            logger.info(f"Sending batch of {len(commands)} commands")
            self.sock.sendall(json.dumps(batch_command).encode('utf-8'))
            response_data = self.receive_full_response()
            response = json.loads(response_data.decode('utf-8'))
            if response.get("status") == "error":
                logger.error(f"Blender batch error: {response.get('message')}")
                raise Exception(response.get("message", "Unknown error"))
            results = response.get("result", [])
            if len(results) != len(commands):
                logger.warning(f"Batch response mismatch: expected {len(commands)} results, got {len(results)}")
            return results
        except Exception as e:
            logger.error(f"Error in batch communication: {str(e)}", exc_info=True)
            self.disconnect()
            raise Exception(f"Connection to Blender lost: {str(e)}")

# Global connection management
_blender_connection = None

def get_blender_connection(host: str = "localhost", port: int = 9876) -> BlenderConnection:
    """Get or create a persistent Blender connection"""
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
    """Manage server startup and shutdown lifecycle asynchronously"""
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
    description="Blender integration through the Model Context Protocol",
    lifespan=server_lifespan
)

# Basic tools
@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Get detailed information about the current Blender scene"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_scene_info")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting scene info: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def get_object_info(ctx: Context, object_name: str) -> str:
    """Get detailed information about a specific object"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_object_info", {"name": object_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting object info: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def create_object(
    ctx: Context,
    type: str = "CUBE",
    name: str = None,
    location: List[float] = None,
    rotation: List[float] = None,
    scale: List[float] = None,
    color: List[float] = None
) -> str:
    """Create a new object in the Blender scene"""
    try:
        blender = get_blender_connection()
        loc = location or [0, 0, 0]
        rot = rotation or [0, 0, 0]
        sc = scale or [1, 1, 1]
        params = {"type": type, "location": loc, "rotation": rot, "scale": sc}
        if name:
            params["name"] = name

        # Prepare batch commands if color is specified
        commands = [{"type": "create_object", "params": params}]
        if color:
            color_params = {"object_name": name or "unnamed", "color": color}
            commands.append({"type": "set_material", "params": color_params})

        # Use batch if there's more than one command
        if len(commands) > 1:
            results = blender.send_batch(commands)
            created_obj = results[0].get("name", "unnamed")
            return f"Created {type} object: {created_obj} with color {color}"
        else:
            result = blender.send_command("create_object", params)
            return f"Created {type} object: {result['name']}"
    except Exception as e:
        logger.error(f"Error creating object: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def modify_object(
    ctx: Context,
    name: str,
    location: List[float] = None,
    rotation: List[float] = None,
    scale: List[float] = None,
    visible: bool = None
) -> str:
    """Modify an existing object"""
    try:
        blender = get_blender_connection()
        params = {"name": name}
        if location:
            params["location"] = location
        if rotation:
            params["rotation"] = rotation
        if scale:
            params["scale"] = scale
        if visible is not None:
            params["visible"] = visible
        result = blender.send_command("modify_object", params)
        return f"Modified object: {result['name']}"
    except Exception as e:
        logger.error(f"Error modifying object: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def delete_object(ctx: Context, name: str) -> str:
    """Delete an object from the scene"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("delete_object", {"name": name})
        return f"Deleted object: {name}"
    except Exception as e:
        logger.error(f"Error deleting object: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def set_material(
    ctx: Context,
    object_name: str,
    material_name: str = None,
    color: List[float] = None
) -> str:
    """Set or create a material for an object"""
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name}
        if material_name:
            params["material_name"] = material_name
        if color:
            params["color"] = color
        result = blender.send_command("set_material", params)
        return f"Applied material to {object_name}: {result.get('material_name', 'unknown')}"
    except Exception as e:
        logger.error(f"Error setting material: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def create_keyframe(
    ctx: Context,
    object_name: str,
    frame: int,
    location: List[float] = None,
    rotation: List[float] = None,
    scale: List[float] = None
) -> str:
    """Add a keyframe for an object at a specific frame"""
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name, "frame": frame}
        if location:
            params["location"] = location
        if rotation:
            params["rotation"] = rotation
        if scale:
            params["scale"] = scale
        result = blender.send_command("create_keyframe", params)
        return f"Added keyframe for {object_name} at frame {frame}"
    except Exception as e:
        logger.error(f"Error adding keyframe: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def create_terrain(
    ctx: Context,
    name: str = "Terrain",
    size: List[float] = None,
    height: float = 1.0,
    subdivisions: int = 64
) -> str:
    """Create a procedural terrain"""
    try:
        blender = get_blender_connection()
        size = size or [10.0, 10.0]
        params = {"name": name, "size": size, "height": height, "subdivisions": subdivisions}
        result = blender.send_command("create_terrain", params)
        return f"Created terrain {name} with size {size} and height {height}"
    except Exception as e:
        logger.error(f"Error creating terrain: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def export_asset(
    ctx: Context,
    object_name: str,
    export_path: str,
    format: str = "GLTF"  # Options: GLTF, FBX
) -> str:
    """Export an object as a game or web asset"""
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name, "export_path": export_path, "format": format.upper()}
        result = blender.send_command("export_asset", params)
        return f"Exported {object_name} to {export_path} as {format}"
    except Exception as e:
        logger.error(f"Error exporting asset: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def execute_blender_code(ctx: Context, code: str) -> str:
    """Execute arbitrary Python code in Blender"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("execute_code", {"code": code})
        return f"Code executed: {result.get('result', '')}"
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return f"Error: {str(e)}"

@mcp.prompt()
def create_basic_object() -> str:
    """Create a single object with basic properties"""
    return """Create a blue cube at position [0, 1, 0]"""

@mcp.prompt()
def modify_basic_object() -> str:
    """Modify a single property of an object"""
    return """Make the cube red"""

def main():
    """Run the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()