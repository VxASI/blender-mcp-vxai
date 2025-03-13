from mcp.server.fastmcp import FastMCP, Context
import socket
import json
import logging
import time
import os
import base64
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

# Configure logging
LOG_DIR = "/tmp"  # Adjust this path as needed for your system
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
    timeout: float = 15.0
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
        try:
            while True:
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

    def send_batch(self, commands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Send a batch of commands to Blender."""
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
    description="Blender integration through the Model Context Protocol with advanced prebuilt tools",
    lifespan=server_lifespan
)

# Original Tools (Preserved)
@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Get detailed information about the current Blender scene."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_scene_info")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting scene info: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def create_object(
    ctx: Context,
    type: str = "CUBE",
    name: str = None,
    location: List[float] = None,
    rotation: List[float] = None,
    scale: List[float] = None,
    relative_to: str = None
) -> str:
    """Create a new object in the Blender scene."""
    try:
        blender = get_blender_connection()
        params = {"type": type}
        if name:
            params["name"] = name
        if location:
            params["location"] = location
        if rotation:
            params["rotation"] = rotation
        if scale:
            params["scale"] = scale
        if relative_to:
            params["relative_to"] = relative_to
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
    """Modify an existing object."""
    try:
        blender = get_blender_connection()
        params = {"name": name}
        if location is not None:
            params["location"] = location
        if rotation is not None:
            params["rotation"] = rotation
        if scale is not None:
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
    """Delete an object from the scene."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("delete_object", {"name": name})
        return f"Deleted object: {result['deleted']}"
    except Exception as e:
        logger.error(f"Error deleting object: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def set_material(
    ctx: Context,
    object_name: str,
    material_name: str = None,
    color: List[float] = None,
    material_type: str = "DIFFUSE",
    metallic: float = None,
    roughness: float = None,
    ior: float = None
) -> str:
    """Set or create a material for an object."""
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name, "material_type": material_type}
        if material_name:
            params["material_name"] = material_name
        if color:
            params["color"] = color
        if metallic is not None:
            params["metallic"] = metallic
        if roughness is not None:
            params["roughness"] = roughness
        if ior is not None:
            params["ior"] = ior
        result = blender.send_command("set_material", params)
        return f"Applied material to {object_name}: {result['material_name']}"
    except Exception as e:
        logger.error(f"Error setting material: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def subdivide_mesh(ctx: Context, object_name: str, cuts: int) -> str:
    """Subdivide the mesh of an object."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("subdivide_mesh", {"name": object_name, "cuts": cuts})
        return f"Subdivided {object_name} with {cuts} cuts"
    except Exception as e:
        logger.error(f"Error subdividing mesh: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def add_modifier(ctx: Context, object_name: str, modifier_type: str, params: Dict[str, Any]) -> str:
    """Add a modifier to an object."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("add_modifier", {"name": object_name, "modifier_type": modifier_type, "params": params})
        return f"Added {modifier_type} modifier to {object_name}"
    except Exception as e:
        logger.error(f"Error adding modifier: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def apply_modifier(ctx: Context, object_name: str, modifier_name: str) -> str:
    """Apply a modifier to an object."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("apply_modifier", {"name": object_name, "modifier_name": modifier_name})
        return f"Applied {modifier_name} to {object_name}"
    except Exception as e:
        logger.error(f"Error applying modifier: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def boolean_operation(ctx: Context, obj1: str, obj2: str, operation: str) -> str:
    """Perform a boolean operation between two objects."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("boolean_operation", {"obj1": obj1, "obj2": obj2, "operation": operation})
        return f"Performed {operation} boolean operation on {obj1} with {obj2}"
    except Exception as e:
        logger.error(f"Error performing boolean operation: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def select_faces_by_normal(ctx: Context, object_name: str, normal: List[float], tolerance: float) -> str:
    """Select faces of an object based on their normal direction."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("select_faces_by_normal", {"name": object_name, "normal": normal, "tolerance": tolerance})
        return f"Selected {result['selected_faces']} faces on {object_name}"
    except Exception as e:
        logger.error(f"Error selecting faces: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def extrude_selected_faces(ctx: Context, object_name: str, distance: float) -> str:
    """Extrude the selected faces of an object."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("extrude_selected_faces", {"name": object_name, "distance": distance})
        return f"Extruded selected faces on {object_name} by {distance}"
    except Exception as e:
        logger.error(f"Error extruding faces: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def create_camera(
    ctx: Context,
    name: str,
    location: List[float],
    rotation: List[float],
    focal_length: float
) -> str:
    """Create a new camera in the scene."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_camera", {"name": name, "location": location, "rotation": rotation, "focal_length": focal_length})
        return f"Created camera: {result['created_camera']}"
    except Exception as e:
        logger.error(f"Error creating camera: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def set_active_camera(ctx: Context, name: str) -> str:
    """Set a camera as the active camera for rendering."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("set_active_camera", {"name": name})
        return f"Set active camera: {result['set_active_camera']}"
    except Exception as e:
        logger.error(f"Error setting active camera: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def render_scene(ctx: Context, filepath: str, resolution: List[int]) -> str:
    """Render the scene to an image file."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("render_scene", {"filepath": filepath, "resolution": resolution})
        return f"Rendered scene to {result['rendered_to']}"
    except Exception as e:
        logger.error(f"Error rendering scene: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
def get_render_preview(ctx: Context, resolution: List[int] = [200, 200]) -> Dict[str, str]:
    """Get a preview render of the current scene, save it server-side, and return file path and base64."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_render_preview", {"resolution": resolution})
        base64_string = result["image_base64"]
        
        preview_path = os.path.join(LOG_DIR, "render_preview.png")
        with open(preview_path, "wb") as f:
            f.write(base64.b64decode(base64_string))
        logger.info(f"Preview image saved to {preview_path}")
        
        return {
            "file_path": preview_path,
            # "base64_string": base64_string,
            "message": f"Preview rendered and saved to {preview_path}. Please describe what you see if adjustments are needed."
        }
    except Exception as e:
        logger.error(f"Error getting render preview: {str(e)}")
        return {"error": f"Error: {str(e)}"}

@mcp.tool()
def point_camera_at(ctx: Context, camera_name: str, target_location: List[float]) -> str:
    """Point a camera at a specific target location in the scene."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("point_camera_at", {
            "camera_name": camera_name,
            "target_location": target_location
        })
        return f"Pointed {camera_name} at {target_location}"
    except Exception as e:
        logger.error(f"Error pointing camera: {str(e)}")
        return f"Error: {str(e)}"

# New Prebuilt Modular Tool for Car
@mcp.tool()
def create_car(
    ctx: Context,
    name: str = "Car",
    body_scale: List[float] = [2.0, 1.0, 0.5],  # Length, Width, Height
    hood_scale: List[float] = [0.8, 0.9, 0.2],  # Length, Width, Height
    tire_radius: float = 0.4,
    tire_thickness: float = 0.2,
    color: List[float] = [0.8, 0.2, 0.2],  # RGB for body
    has_spoiler: bool = False,
    spoiler_scale: List[float] = [0.1, 1.0, 0.1],
    gun_count: int = 0,  # Number of guns
    gun_type: str = "machine_gun",  # "machine_gun" or "cannon"
    gun_position: str = "roof"  # "roof" or "hood"
) -> str:
    """
    Create a modular, realistic car with customizable components.

    Parameters:
    - name: Name of the car (default: "Car").
    - body_scale: [length, width, height] of the car body (default: [2.0, 1.0, 0.5]).
    - hood_scale: [length, width, height] of the hood (default: [0.8, 0.9, 0.2]).
    - tire_radius: Radius of tires (default: 0.4).
    - tire_thickness: Thickness of tires (default: 0.2).
    - color: [R, G, B] values for the body (0.0 to 1.0, default: [0.8, 0.2, 0.2] for red).
    - has_spoiler: Add a spoiler (default: False).
    - spoiler_scale: [length, width, height] of the spoiler (default: [0.1, 1.0, 0.1]).
    - gun_count: Number of guns to add (0-4, default: 0).
    - gun_type: Type of gun ("machine_gun" or "cannon", default: "machine_gun").
    - gun_position: Where to place guns ("roof" or "hood", default: "roof").

    Returns:
    Confirmation string with the car’s name.

    Examples:
    - Simple sedan: create_car(name="Sedan")
    - Sports car with spoiler: create_car(name="SportsCar", body_scale=[1.8, 0.8, 0.4], has_spoiler=True, color=[0.0, 0.0, 1.0])
    - Armored car: create_car(name="TankCar", body_scale=[2.5, 1.2, 0.8], gun_count=2, gun_type="cannon", gun_position="hood")
    """
    try:
        blender = get_blender_connection()
        params = {
            "name": name,
            "body_scale": body_scale,
            "hood_scale": hood_scale,
            "tire_radius": tire_radius,
            "tire_thickness": tire_thickness,
            "color": color,
            "has_spoiler": has_spoiler,
            "spoiler_scale": spoiler_scale,
            "gun_count": min(max(gun_count, 0), 4),  # Clamp between 0 and 4
            "gun_type": gun_type,
            "gun_position": gun_position
        }
        result = blender.send_command("create_car", params)
        return f"Created car: {result['created']}"
    except Exception as e:
        logger.error(f"Error creating car: {str(e)}")
        return f"Error: {str(e)}"

def main():
    """Run the FastMCP server."""
    mcp.run()

if __name__ == "__main__":
    main()