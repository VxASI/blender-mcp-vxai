#!/usr/bin/env python3
"""
Blender MCP Server - A Model Context Protocol server for Blender 3D
This server enables AI assistants (e.g., Cursor) to interact with Blender through text commands
via stdio transport, while connecting to the Blender addon via sockets.
"""

import asyncio
import json
import logging
import socket
import time
import os
import tempfile
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Union, AsyncIterator

from mcp.server.fastmcp import FastMCP, Context, Image
import base64
from io import BytesIO
from PIL import Image as PILImage, ImageDraw

# Configure logging
logger = logging.getLogger("BlenderMCP")
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler (with error handling)
log_file = os.path.join(tempfile.gettempdir(), "blender_mcp.log")
try:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")
except Exception as e:
    logger.warning(f"Failed to set up file logging: {str(e)}. Falling back to console logging.")

# Connection configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876
DEFAULT_TIMEOUT = 15.0  # seconds
RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 2.0  # seconds

@dataclass
class BlenderConnection:
    """Manages socket connection to the Blender addon"""
    host: str
    port: int
    sock: Optional[socket.socket] = None
    timeout: float = DEFAULT_TIMEOUT
    _last_command_time: float = field(default_factory=time.time)
    
    def connect(self, attempts: int = RECONNECT_ATTEMPTS) -> bool:
        """
        Connect to the Blender addon socket server
        """
        if self.sock and self._check_connection():
            return True
            
        self.disconnect()
        
        for attempt in range(1, attempts + 1):
            try:
                logger.info(f"Connecting to Blender at {self.host}:{self.port} (attempt {attempt}/{attempts})")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                self._last_command_time = time.time()
                
                ping_response = self.send_command("ping")
                if ping_response and ping_response.get("pong"):
                    logger.info(f"Connected to Blender at {self.host}:{self.port}")
                    return True
                else:
                    logger.warning("Connection established but ping failed")
                    self.disconnect()
                    
            except (socket.timeout, ConnectionRefusedError) as e:
                logger.warning(f"Connection attempt {attempt} failed: {str(e)}")
                if self.sock:
                    self.sock.close()
                    self.sock = None
                
                if attempt < attempts:
                    logger.info(f"Retrying in {RECONNECT_DELAY} seconds...")
                    time.sleep(RECONNECT_DELAY)
            except Exception as e:
                logger.error(f"Unexpected error connecting to Blender: {str(e)}")
                if self.sock:
                    self.sock.close()
                    self.sock = None
                break
                
        logger.error(f"Failed to connect to Blender after {attempts} attempts")
        return False
    
    def _check_connection(self) -> bool:
        """
        Check if the current connection is still alive
        """
        if not self.sock:
            return False
            
        time_since_last = time.time() - self._last_command_time
        if time_since_last > 300:  # 5 minutes
            logger.info(f"Connection idle for {time_since_last:.1f}s, testing it")
            try:
                self.sock.settimeout(5.0)
                ping_cmd = json.dumps({"type": "ping"}).encode('utf-8')
                self.sock.sendall(ping_cmd)
                response = self.receive_data(self.sock, 1024)
                response_obj = json.loads(response.decode('utf-8'))
                
                if response_obj.get("result", {}).get("pong"):
                    logger.debug("Connection is alive")
                    self._last_command_time = time.time()
                    return True
                else:
                    logger.warning("Connection test failed, no proper response")
                    return False
            except Exception as e:
                logger.warning(f"Connection test failed: {str(e)}")
                return False
            finally:
                if self.sock:
                    self.sock.settimeout(self.timeout)
        
        return True
    
    def disconnect(self) -> None:
        """Disconnect from the Blender addon"""
        if self.sock:
            try:
                self.sock.close()
                logger.info("Disconnected from Blender")
            except Exception as e:
                logger.error(f"Error disconnecting from Blender: {str(e)}")
            finally:
                self.sock = None

    def receive_data(self, sock: socket.socket, buffer_size: int = 8192) -> bytes:
        """
        Receive complete JSON response from socket
        """
        chunks = []
        start_time = time.time()
        timeout = self.timeout
        sock.settimeout(timeout)
        
        while True:
            try:
                chunk = sock.recv(buffer_size)
                if not chunk:
                    if not chunks:
                        raise ConnectionError("Connection closed before receiving any data")
                    break
                
                chunks.append(chunk)
                data = b''.join(chunks)
                
                try:
                    json.loads(data.decode('utf-8'))
                    return data
                except json.JSONDecodeError:
                    elapsed = time.time() - start_time
                    if elapsed > timeout * 0.8:
                        remaining = max(timeout * 0.5, 2.0)
                        logger.debug(f"Extending receive timeout by {remaining:.1f}s")
                        sock.settimeout(remaining)
                    continue
                    
            except socket.timeout:
                if chunks:
                    logger.warning("Socket timeout during receive, trying to parse partial data")
                    data = b''.join(chunks)
                    try:
                        json.loads(data.decode('utf-8'))
                        return data
                    except json.JSONDecodeError:
                        raise TimeoutError("Incomplete JSON response after timeout")
                else:
                    raise TimeoutError("Timeout waiting for response, no data received")
        
        data = b''.join(chunks)
        try:
            json.loads(data.decode('utf-8'))
            return data
        except json.JSONDecodeError:
            raise ValueError("Incomplete JSON response received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a command to Blender and return the response
        """
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Blender. Is the addon running?")
        
        self._last_command_time = time.time()
        
        command = {
            "type": command_type,
            "params": params or {}
        }
        
        try:
            logger.info(f"Sending command: {command_type}")
            logger.debug(f"Command details: {json.dumps(command)}")
            
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            self.sock.settimeout(self.timeout)
            response_data = self.receive_data(self.sock)
            
            response = json.loads(response_data.decode('utf-8'))
            logger.debug(f"Response: {json.dumps(response)}")
            
            if response.get("status") == "error":
                error_msg = response.get("message", "Unknown error from Blender")
                logger.error(f"Blender error: {error_msg}")
                return {"error": error_msg}
            
            return response.get("result", {})
            
        except socket.timeout:
            logger.error("Socket timeout while waiting for response from Blender")
            self.disconnect()
            raise TimeoutError("Timeout waiting for Blender response")
            
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.disconnect()
            raise ConnectionError(f"Connection to Blender lost: {str(e)}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Blender: {str(e)}")
            if 'response_data' in locals():
                logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
            raise ValueError(f"Invalid response from Blender: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error communicating with Blender: {str(e)}")
            self.disconnect()
            raise

# Singleton connection manager
_blender_connection = None

def get_blender_connection(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> BlenderConnection:
    """
    Get or create a persistent Blender connection
    """
    global _blender_connection
    
    if _blender_connection is None:
        _blender_connection = BlenderConnection(host=host, port=port)
    
    if not _blender_connection.connect():
        _blender_connection = None
        raise ConnectionError("Could not connect to Blender. Is the addon running?")
    
    return _blender_connection

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """
    Manage server startup and shutdown lifecycle
    """
    try:
        logger.info("BlenderMCP server starting up")
        try:
            connection = get_blender_connection()
            version_info = connection.send_command("get_version_info")
            logger.info(f"Connected to Blender {version_info.get('version', 'unknown')}")
        except Exception as e:
            logger.error(f"Blender connection failed: {str(e)}")
            raise  # Fail if Blender connection is not established
        yield {}
    finally:
        global _blender_connection
        if _blender_connection:
            logger.info("Disconnecting from Blender")
            _blender_connection.disconnect()
            _blender_connection = None
        logger.info("BlenderMCP server shutdown complete")

# Create MCP server
mcp = FastMCP(
    "BlenderMCP",
    description="Blender integration through the Model Context Protocol",
    lifespan=server_lifespan
)

# Resource endpoints (JSON data)
@mcp.resource("scene://info")
def scene_info() -> str:
    """Get information about the current Blender scene"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_scene_info")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting scene info: {str(e)}")
        return json.dumps({"error": str(e)})

@mcp.resource("scene://objects")
def object_list() -> str:
    """Get list of objects in the Blender scene"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_object_list")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting object list: {str(e)}")
        return json.dumps({"error": str(e)})

@mcp.resource("scene://materials")
def material_list() -> str:
    """Get list of materials in the Blender scene"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_material_list")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting material list: {str(e)}")
        return json.dumps({"error": str(e)})

@mcp.resource("scene://preview")
def render_preview() -> str:
    """Get a preview render of the current Blender scene as base64-encoded PNG"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("render_preview", {"format": "PNG"})
        
        if "image_data" in result:
            return result["image_data"]  # Return base64 string directly
        else:
            img = PILImage.new('RGB', (400, 300), color=(64, 64, 64))
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), "Error getting preview render", fill=(255, 255, 255))
            if "error" in result:
                draw.text((20, 50), result["error"], fill=(255, 100, 100))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
    except Exception as e:
        logger.error(f"Error getting preview render: {str(e)}")
        img = PILImage.new('RGB', (400, 300), color=(64, 64, 64))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Error", fill=(255, 255, 255))
        draw.text((20, 50), str(e), fill=(255, 100, 100))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Tool endpoints for manipulating Blender
@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Get detailed information about the current Blender scene"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_scene_info")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting scene info: {str(e)}"

@mcp.tool()
def get_object_info(ctx: Context, object_name: str) -> str:
    """
    Get detailed information about a specific object in the Blender scene
    
    Parameters:
    - object_name: The name of the object to get information about
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_object_info", {"name": object_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting object info: {str(e)}"

@mcp.tool()
def create_primitive(
    ctx: Context,
    type: str = "CUBE",
    name: str = None,
    location: List[float] = None,
    color: List[float] = None
) -> str:
    """
    Create a basic primitive object in Blender
    
    Parameters:
    - type: Object type (CUBE, SPHERE, CYLINDER, PLANE, CONE, TORUS)
    - name: Optional custom name for the object
    - location: Optional [x, y, z] location coordinates
    - color: Optional [R, G, B] color values (0.0-1.0)
    """
    try:
        blender = get_blender_connection()
        params = {"type": type}
        
        if name:
            params["name"] = name
        if location:
            params["location"] = location
        
        result = blender.send_command("create_object", params)
        object_name = result.get("name", "unknown")
        
        if color:
            blender.send_command("set_material", {
                "object_name": object_name,
                "color": color
            })
        
        response = f"Created {type}"
        if name:
            response += f" named '{object_name}'"
        if location:
            response += f" at location {location}"
        if color:
            response += f" with color {color}"
            
        return response
        
    except Exception as e:
        return f"Error creating primitive: {str(e)}"

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
    """
    Create a new object in the Blender scene with full parameters
    
    Parameters:
    - type: Object type (CUBE, SPHERE, CYLINDER, PLANE, CONE, TORUS, EMPTY, CAMERA, LIGHT)
    - name: Optional name for the object
    - location: Optional [x, y, z] location coordinates
    - rotation: Optional [x, y, z] rotation in radians
    - scale: Optional [x, y, z] scale factors
    - color: Optional [R, G, B] color values (0.0-1.0)
    """
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
            
        result = blender.send_command("create_object", params)
        object_name = result.get("name", "unknown")
        
        if color:
            material_result = blender.send_command("set_material", {
                "object_name": object_name,
                "color": color
            })
            material_name = material_result.get("material_name", "unknown")
        
        response_parts = [f"Created {type}"]
        if name:
            response_parts.append(f"named '{object_name}'")
        if location:
            response_parts.append(f"at {location}")
        if rotation:
            response_parts.append(f"with rotation {rotation}")
        if scale:
            response_parts.append(f"and scale {scale}")
        if color:
            response_parts.append(f"with color {color}")
            
        return " ".join(response_parts)
        
    except Exception as e:
        return f"Error creating object: {str(e)}"

@mcp.tool()
def modify_object(
    ctx: Context,
    name: str,
    location: List[float] = None,
    rotation: List[float] = None,
    scale: List[float] = None,
    visible: bool = None,
    color: List[float] = None
) -> str:
    """
    Modify an existing object in the Blender scene
    
    Parameters:
    - name: Name of the object to modify
    - location: Optional [x, y, z] location coordinates
    - rotation: Optional [x, y, z] rotation in radians
    - scale: Optional [x, y, z] scale factors
    - visible: Optional boolean to set visibility
    - color: Optional [R, G, B] color values to change material color
    """
    try:
        blender = get_blender_connection()
        
        params = {"name": name}
        changes = []
        
        if location is not None:
            params["location"] = location
            changes.append(f"location to {location}")
            
        if rotation is not None:
            params["rotation"] = rotation
            changes.append(f"rotation to {rotation}")
            
        if scale is not None:
            params["scale"] = scale
            changes.append(f"scale to {scale}")
            
        if visible is not None:
            params["visible"] = visible
            changes.append(f"visibility to {visible}")
        
        if len(params) > 1:
            result = blender.send_command("modify_object", params)
        
        if color is not None:
            material_result = blender.send_command("set_material", {
                "object_name": name,
                "color": color
            })
            changes.append(f"color to {color}")
        
        if changes:
            return f"Modified object '{name}': set " + ", ".join(changes)
        else:
            return f"No changes made to object '{name}'"
            
    except Exception as e:
        return f"Error modifying object: {str(e)}"

@mcp.tool()
def delete_object(ctx: Context, name: str) -> str:
    """
    Delete an object from the Blender scene
    
    Parameters:
    - name: Name of the object to delete
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("delete_object", {"name": name})
        
        if "deleted" in result:
            return f"Deleted object: {result['deleted']}"
        else:
            return f"Object '{name}' deleted successfully"
            
    except Exception as e:
        return f"Error deleting object: {str(e)}"

@mcp.tool()
def set_material(
    ctx: Context,
    object_name: str,
    material_name: str = None,
    color: List[float] = None,
    metallic: float = None,
    roughness: float = None,
    specular: float = None
) -> str:
    """
    Set or create a material for an object
    
    Parameters:
    - object_name: Name of the object to apply the material to
    - material_name: Optional name of the material to use or create
    - color: Optional [R, G, B] color values (0.0-1.0)
    - metallic: Optional metallic factor (0.0-1.0)
    - roughness: Optional roughness factor (0.0-1.0)
    - specular: Optional specular intensity (0.0-1.0)
    """
    try:
        blender = get_blender_connection()
        
        params = {"object_name": object_name}
        changes = []
        
        if material_name:
            params["material_name"] = material_name
            changes.append(f"material to '{material_name}'")
            
        if color:
            params["color"] = color
            changes.append(f"color to {color}")
            
        if metallic is not None:
            params["metallic"] = metallic
            changes.append(f"metallic to {metallic}")
            
        if roughness is not None:
            params["roughness"] = roughness
            changes.append(f"roughness to {roughness}")
            
        if specular is not None:
            params["specular"] = specular
            changes.append(f"specular to {specular}")
        
        result = blender.send_command("set_material", params)
        
        if result.get("material_name"):
            return f"Applied material '{result['material_name']}' to object '{object_name}' with " + ", ".join(changes)
        else:
            return f"Set material properties on object '{object_name}': " + ", ".join(changes)
            
    except Exception as e:
        return f"Error setting material: {str(e)}"

@mcp.tool()
def render_scene(
    ctx: Context, 
    width: int = 800, 
    height: int = 600,
    output_path: str = None
) -> str:
    """
    Render the current Blender scene
    
    Parameters:
    - width: Render width in pixels
    - height: Render height in pixels
    - output_path: Optional filepath to save the rendered image
    """
    try:
        blender = get_blender_connection()
        
        params = {
            "resolution_x": width,
            "resolution_y": height
        }
        
        if output_path:
            params["output_path"] = output_path
        
        result = blender.send_command("render_scene", params)
        
        if result.get("rendered"):
            response = f"Rendered scene at {width}x{height}"
            if output_path and result.get("output_path"):
                response += f", saved to {result['output_path']}"
            return response
        else:
            return "Failed to render scene"
            
    except Exception as e:
        return f"Error rendering scene: {str(e)}"

@mcp.tool()
def execute_blender_code(ctx: Context, code: str) -> str:
    """
    Execute arbitrary Python code in Blender
    
    Parameters:
    - code: The Python code to execute in Blender
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("execute_code", {"code": code})
        
        if result.get("executed"):
            return "Code executed successfully in Blender"
        elif result.get("result"):
            return f"Code executed with result: {result['result']}"
        else:
            return "Code execution failed"
            
    except Exception as e:
        return f"Error executing code: {str(e)}"

@mcp.tool()
def create_text_object(
    ctx: Context,
    text: str,
    name: str = None,
    location: List[float] = None,
    extrude: float = 0.0,
    bevel_depth: float = 0.0,
    color: List[float] = None
) -> str:
    """
    Create a 3D text object in Blender
    
    Parameters:
    - text: The text content
    - name: Optional name for the text object
    - location: Optional [x, y, z] location coordinates
    - extrude: Optional extrusion depth for 3D effect
    - bevel_depth: Optional bevel depth for rounded edges
    - color: Optional [R, G, B] color values (0.0-1.0)
    """
    try:
        blender = get_blender_connection()
        
        params = {
            "text": text
        }
        
        if name:
            params["name"] = name
        if location:
            params["location"] = location
        if extrude is not None:
            params["extrude"] = extrude
        if bevel_depth is not None:
            params["bevel_depth"] = bevel_depth
        
        result = blender.send_command("create_text", params)
        object_name = result.get("name", "unknown")
        
        if color:
            blender.send_command("set_material", {
                "object_name": object_name,
                "color": color
            })
        
        response = f"Created text object '{text}'"
        if name:
            response += f" named '{object_name}'"
        if location:
            response += f" at {location}"
        if extrude > 0:
            response += f" with extrusion {extrude}"
        if bevel_depth > 0:
            response += f" and bevel {bevel_depth}"
        if color:
            response += f" in color {color}"
        
        return response
            
    except Exception as e:
        return f"Error creating text object: {str(e)}"

@mcp.tool()
def add_animation(
    ctx: Context,
    object_name: str,
    property: str,
    keyframes: List[Dict[str, Any]],
    interpolation: str = "BEZIER"
) -> str:
    """
    Add keyframe animation to an object property
    
    Parameters:
    - object_name: Name of the object to animate
    - property: Property to animate (location, rotation, scale)
    - keyframes: List of keyframes with format [{"frame": frame_number, "value": property_value}, ...]
    - interpolation: Animation interpolation type (BEZIER, LINEAR, CONSTANT)
    """
    try:
        blender = get_blender_connection()
        
        params = {
            "object_name": object_name,
            "property": property,
            "keyframes": keyframes,
            "interpolation": interpolation
        }
        
        result = blender.send_command("add_animation", params)
        
        if result.get("keyframes_added"):
            return f"Added {result['keyframes_added']} keyframes to {object_name}.{property} with {interpolation} interpolation"
        else:
            return f"Failed to add animation to {object_name}.{property}"
            
    except Exception as e:
        return f"Error adding animation: {str(e)}"

@mcp.tool()
def import_model(
    ctx: Context,
    file_path: str,
    location: List[float] = None,
    rotation: List[float] = None,
    scale: float = None
) -> str:
    """
    Import a 3D model file into the Blender scene
    
    Parameters:
    - file_path: Path to the model file (must be accessible to Blender)
    - location: Optional [x, y, z] placement location
    - rotation: Optional [x, y, z] rotation in radians
    - scale: Optional uniform scale factor
    """
    try:
        blender = get_blender_connection()
        
        params = {
            "file_path": file_path
        }
        
        if location:
            params["location"] = location
        if rotation:
            params["rotation"] = rotation
        if scale is not None:
            params["scale"] = scale
        
        result = blender.send_command("import_model", params)
        
        if result.get("imported"):
            object_count = result.get("object_count", 1)
            model_name = result.get("model_name", file_path)
            
            response = f"Imported {model_name}"
            if object_count > 1:
                response += f" containing {object_count} objects"
            if location:
                response += f" at location {location}"
            if rotation:
                response += f" with rotation {rotation}"
            if scale:
                response += f" scaled by {scale}"
                
            return response
        else:
            return f"Failed to import model from {file_path}"
            
    except Exception as e:
        return f"Error importing model: {str(e)}"

# Example prompts
@mcp.prompt()
def create_basic_scene() -> str:
    """Create a simple scene with basic objects"""
    return """Create a blue cube at position [0, 1, 0] and a red sphere at position [2, 0, 0]"""

@mcp.prompt()
def create_text_scene() -> str:
    """Create a 3D text with materials"""
    return """Create a 3D text object saying "Hello Blender" with metallic gold material"""

@mcp.prompt()
def create_animation() -> str:
    """Create an animated scene"""
    return """Create a cube that moves from position [0,0,0] to [5,0,0] over 30 frames"""

def main():
    """Run the BlenderMCP server with stdio transport for Cursor"""
    try:
        logger.info("Starting BlenderMCP server for Cursor integration...")
        mcp.run()  # Use stdio transport
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    main()