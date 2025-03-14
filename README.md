# Blender MCP VXAI 

Pro Tip: You can use this to ask your agent to export the 3d model directly into the project you are working on to use them in your app instantly. [Demo](https://youtu.be/sHRI0nPan20?feature=shared)
## Description

Blender MCP VXAI a powerful integration that allows you to control Blender using natural language through MCP Clients. This tool enables you to create, modify, and manipulate 3D models, animations, and scenes in Blender by simply describing what you want to do. It bridges the gap between AI language models and 3D creation, making Blender more accessible and efficient for both beginners and experienced users. This is a simple tool where the agent can let the AI agent create scripts for you while getting feedback and building the exact scene you want.

## Features

- Control Blender using natural language commands
- Seamless integration with MCP Clients
- Automate complex Blender operations with simple text instructions
- Enhance your 3D workflow with AI assistance

## Installation

### Prerequisites

- Blender 
- Python 3.8+

### Step 1: Install UV

UV is required to run the MCP server.

**macOS:**
```bash
brew install uv
```

**Windows/Linux:**
```bash
pip install uv
```

### Step 2: Configure Your Environment

#### For Cursor:
1. Click "+ Add new Server"
2. Configure with:
   - Name: `blender-mcp`
   - Command: `uvx --from blender-mcp-vxai start`

#### For Claude Desktop:
1. Go to Claude > Settings > Developer > Edit Config
2. Open `claude_desktop_config.json` and add:
```json
{
    "mcpServers": {
        "blender": {
            "command": "uvx",
            "args": [
                "--from blender-mcp-vxai start"
            ]
        }
    }
}
```

### Step 3: Install the Blender Addon

1. Download `blender_mcp_addon.py`
2. Open Blender
3. Go to Edit > Preferences > Add-ons
4. Click "Install from Disk" and select the `blender_mcp_addon.py` file
5. Enable the addon by checking the box next to "Blender MCP"
6. In Blender, go to the 3D View sidebar (press N if not visible)
7. Find the "BlenderMCP" tab
8. Start MCP server



## Usage


## Available Tools

- **Object Creation**: Create primitives, import models, and generate complex shapes
- **Modeling**: Modify meshes, add modifiers, and sculpt objects
- **Materials**: Create and apply materials, textures, and shaders
- **Animation**: Create keyframes, animate properties, and set up rigging
- **Rendering**: Configure render settings, lighting, and camera positions
- **Scene Management**: Organize objects, collections, and scenes

## Example Usecases
- Share an image and ask it to create a low poly version of it
- Update elements on scene as needed, describe it in detail
- Build simple to complex scenes one step a time
- change camera, color, add lighting etc


## Troubleshooting

- If the connection fails, ensure the MCP server is running
- Check that the addon is properly installed and enabled
- For specific errors, check the Blender console for detailed messages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE


CREDITS TO https://github.com/ahujasid/blender-mcp for the idea!
