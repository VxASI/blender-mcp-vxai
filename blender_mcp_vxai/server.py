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
    timeout: float = 60.0
    max_reconnect_attempts: int = 3
    base_reconnect_delay: float = 1.0

    def connect(self) -> bool:
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
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Blender: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self) -> bytes:
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

_blender_connection = None

def get_blender_connection(host: str = "localhost", port: int = 9876) -> BlenderConnection:
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
    logger.info("BlenderMCP server starting up")
    try:
        get_blender_connection()
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

mcp = FastMCP(
    "BlenderMCP",
    description="Blender integration for dynamic scene manipulation via MCP",
    lifespan=server_lifespan
)

@mcp.tool()
def get_scene_info(
    ctx: Context,
    filters: Dict[str, Any] = None,
    properties: List[str] = None,
    sub_object_data: Dict[str, Any] = None,
    limit: int = None,
    offset: int = 0,
    timeout: float = 5.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Retrieve detailed information about the current Blender scene with advanced filtering and efficiency options.

    Parameters:
        filters (dict, optional): Filters to narrow down objects (e.g., {"type": "MESH"}).
        properties (list, optional): Properties to include (e.g., ["name", "vertices"]).
        sub_object_data (dict, optional): Options for sub-object data (e.g., {"vertices": {"sample_rate": 0.1}}).
        limit (int, optional): Max number of objects to return.
        offset (int, optional): Starting index for pagination.
        timeout (float, optional): Max time in seconds.
        verbose (bool, optional): If True, returns detailed logs.

    Returns:
        dict: Scene data or an error with a suggestion if timed out, including verbose logs if enabled.
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
        if verbose and result.get("status") != "error":
            result["verbose_log"] = f"Processed {len(result.get('objects', []) + result.get('cameras', []) + result.get('lights', []))} objects in {timeout} seconds."
        return result
    except Exception as e:
        logger.error(f"Error getting scene info: {str(e)}")
        return {"error": str(e), "suggestion": "Try applying more specific filters or increasing the timeout."}

@mcp.tool()
def run_script(ctx: Context, script: str, verbose: bool = False) -> str:
    """
    Execute a Python script in Blender to manipulate the scene.

    Parameters:
        script (str): A Python script to execute in Blender.
        verbose (bool, optional): If True, returns detailed execution logs.

    Returns:
        str: Confirmation or error message, with verbose details if enabled.
    """
    try:
        blender = get_blender_connection()
        script_encoded = base64.b64encode(script.encode('utf-8')).decode('ascii')
        result = blender.send_command("run_script", {"script": script_encoded})
        message = f"Script executed successfully: {result.get('message', 'No message returned')}"
        if verbose:
            message += f" | Script length: {len(script)} characters, History entry: {result.get('message', '')[:50]}..."
        return message
    except Exception as e:
        logger.error(f"Error running script: {str(e)}")
        return f"Error: {str(e)} | Suggestion: Check script syntax or ensure object exists."

@mcp.tool()
def edit_mesh(
    ctx: Context,
    object_name: str,
    operation: str,
    parameters: Dict[str, Any],
    verbose: bool = False,
    sequence: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform advanced mesh editing operations on a specified object, with optional sequence of operations.

    Parameters:
        object_name (str): Name of the object to edit.
        operation (str): Operation to perform (e.g., "select_vertices", "extrude").
        parameters (dict): Operation-specific parameters (e.g., {"direction": "normal", "distance": 0.5}).
        verbose (bool, optional): If True, returns detailed logs of each step.
        sequence (list, optional): List of {"operation": str, "parameters": dict} to chain operations.

    Returns:
        dict: Result with "status", "affected_vertices", "message", and "suggest_next_operation" if successful.
    """
    try:
        blender = get_blender_connection()
        params = {
            "object_name": object_name,
            "operation": operation,
            "parameters": parameters,
            "verbose": verbose,
            "sequence": sequence
        }
        result = blender.send_command("edit_mesh", params)
        if verbose and result.get("status") == "success":
            result["verbose_log"] = f"Completed {operation} on {object_name} affecting {result.get('affected_vertices', 0)} vertices."
        return result
    except Exception as e:
        logger.error(f"Error editing mesh: {str(e)}")
        return {"error": str(e), "suggestion": "Ensure object exists and try selecting vertices first if needed."}

@mcp.tool()
def get_tool_examples(ctx: Context) -> str:
    """
    Return detailed examples of how to use all available tools in the MCP server.

    Returns:
        str: A detailed string with examples for each tool and its parameters.
    """
    examples = """
=== Tool Examples for Blender MCP Integration ===

Tool: get_scene_info
  Purpose: Retrieve information about the current Blender scene.
  Recommended Use: Use before editing to inspect objects or vertex data.
  Example 1:
    Call: filters={"type": "MESH"}, properties=["name", "location"], verbose=True
    Description: Gets all mesh objects with names and locations, with verbose logs.
    Expected Output: {"status": "success", "objects": [...], "verbose_log": "Processed X objects..."}
    Next Steps: Use edit_mesh to modify a specific object.
  Example 2:
    Call: filters={"name_contains": "Cube"}, properties=["vertices"], sub_object_data={"vertices": {"sample_rate": 0.1}}, verbose=True
    Description: Samples 10% of a "Cube" object’s vertices with detailed logs.
    Expected Output: {"status": "success", "objects": [{"name": "Cube", "vertices": [...]}], "verbose_log": "..."}
    Next Steps: Analyze data and select vertices for editing.

Tool: run_script
  Purpose: Execute a custom Python script in Blender for complex manipulations.
  Recommended Use: Use for tasks not covered by edit_mesh, like adding objects.
  Example 1:
    Call: script="import bpy; bpy.ops.mesh.primitive_cube_add(size=2); bpy.context.object.name='MyCube'", verbose=True
    Description: Adds a cube and names it "MyCube" with verbose feedback.
    Expected Output: "Script executed successfully: ... | Script length: X characters, History entry: ..."
    Next Steps: Use get_scene_info to verify the new object.
  Example 2:
    Call: script="import bpy; bpy.data.materials.new(name='RedMaterial'); bpy.context.object.data.materials.append(bpy.data.materials['RedMaterial']); bpy.context.object.active_material.diffuse_color=(1,0,0,1)", verbose=True
    Description: Applies a red material to the active object with logs.
    Expected Output: "Script executed successfully: ... | Script length: X characters, History entry: ..."
    Next Steps: Use edit_mesh to shape the object.

Tool: edit_mesh
  Purpose: Modify a mesh with specific operations.
  Recommended Use: Use to reshape objects (e.g., car hood) with targeted edits.
  Operation: select_vertices
    Example 1:
      Call: object_name="Car", operation="select_vertices", parameters={"criteria": {"position_bounds": {"min": [-1,-1,-1], "max": [0,1,1]}}, "selection_mode": "replace", "keep_edit_mode": True}, verbose=True
      Description: Selects vertices within a -1 to 0 range on X, keeping edit mode.
      Expected Output: {"status": "success", "message": "Selected X vertices", "affected_vertices": X, "verbose_log": "Completed select_vertices..."}
      Next Steps: Extrude or scale the selection.
    Example 2:
      Call: object_name="Car", operation="select_vertices", parameters={"criteria": {"vertex_group": "hood"}, "selection_mode": "replace", "keep_edit_mode": True}, verbose=True
      Description: Selects the "hood" vertex group, keeping edit mode.
      Expected Output: {"status": "success", "message": "Selected X vertices", "affected_vertices": X, "verbose_log": "..."}
      Next Steps: Extrude or rotate.
  Operation: extrude
    Example:
      Call: object_name="Car", operation="extrude", parameters={"direction": "normal", "distance": 0.5, "keep_edit_mode": True}, verbose=True
      Description: Extrudes selected vertices upward by 0.5 units.
      Expected Output: {"status": "success", "message": "Extruded with direction normal...", "affected_vertices": Y, "verbose_log": "..."}
      Next Steps: Scale or smooth.
  Operation: scale
    Example:
      Call: object_name="Car", operation="scale", parameters={"values": [1.2, 0.8, 0.5], "pivot_point": "median", "keep_edit_mode": True}, verbose=True
      Description: Widens X by 1.2, narrows Y by 0.8, flattens Z by 0.5.
      Expected Output: {"status": "success", "message": "Scaled Y vertices", "affected_vertices": Y, "verbose_log": "..."}
      Next Steps: Rotate or bevel.
  Operation: rotate
    Example:
      Call: object_name="Car", operation="rotate", parameters={"values": [0, 0.2, 0], "pivot_point": "median", "keep_edit_mode": True}, verbose=True
      Description: Rotates 0.2 radians (11.5°) around Y-axis.
      Expected Output: {"status": "success", "message": "Rotated Y vertices", "affected_vertices": Y, "verbose_log": "..."}
      Next Steps: Smooth or move.
  Operation: move
    Example:
      Call: object_name="Car", operation="move", parameters={"values": [-0.3, 0, 0], "keep_edit_mode": True}, verbose=True
      Description: Moves selected vertices -0.3 units on X.
      Expected Output: {"status": "success", "message": "Moved Y vertices", "affected_vertices": Y, "verbose_log": "..."}
      Next Steps: Extrude or scale.
  Operation: bevel
    Example:
      Call: object_name="Car", operation="bevel", parameters={"offset": 0.1, "segments": 2, "keep_edit_mode": True}, verbose=True
      Description: Bevels selected edges with 0.1 offset and 2 segments.
      Expected Output: {"status": "success", "message": "Beveled X edges", "affected_vertices": Y, "verbose_log": "..."}
      Next Steps: Smooth or subdivide.
  Operation: smooth
    Example:
      Call: object_name="Car", operation="smooth", parameters={"factor": 0.5, "iterations": 2, "keep_edit_mode": False}, verbose=True
      Description: Smooths selected vertices twice with factor 0.5.
      Expected Output: {"status": "success", "message": "Smoothed Y vertices...", "affected_vertices": Y, "verbose_log": "..."}
      Next Steps: Apply curve or subdivide.
  Operation: apply_curve
    Example:
      Call: object_name="Car", operation="apply_curve", parameters={"curve_type": "bezier", "control_points": [[0,0,0], [1,1,0], [2,0,0]], "keep_edit_mode": False}, verbose=True
      Description: Deforms along a Bezier curve with 3 points.
      Expected Output: {"status": "success", "message": "Applied bezier curve...", "affected_vertices": Y, "verbose_log": "..."}
      Next Steps: Smooth or scale.
  Operation: subdivide
    Example:
      Call: object_name="Car", operation="subdivide", parameters={"cuts": 2, "keep_edit_mode": False}, verbose=True
      Description: Subdivides selected faces with 2 cuts.
      Expected Output: {"status": "success", "message": "Subdivided with 2 cuts", "affected_vertices": Y, "verbose_log": "..."}
      Next Steps: Smooth or bevel.
  Sequence Example:
    Call: object_name="Car", operation="select_vertices", parameters={"criteria": {"position_bounds": {"min": [-1,-1,-1], "max": [0,1,1]}}, "keep_edit_mode": True}, sequence=[{"operation": "extrude", "parameters": {"direction": "normal", "distance": 0.5}}, {"operation": "scale", "parameters": {"values": [1.2, 0.8, 0.5], "pivot_point": "median"}}], verbose=True
    Description: Selects vertices, extrudes them, then scales in one call.
    Expected Output: {"status": "success", "message": "Completed sequence...", "affected_vertices": Y, "verbose_log": "..."}
    Next Steps: Rotate or smooth.

=== End of Examples ===

Use these examples to decide which tool to use. Start with get_scene_info to gather data, use edit_mesh for modifications (with sequence for chaining), and use run_script for custom scripts.
"""
    return examples

def main():
    mcp.run()

if __name__ == "__main__":
    main()