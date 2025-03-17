# Blender MCP VXAI

Pro Tip: You can use this to ask your agent to export the 3d model directly into the project you are working on to use them in your app instantly. [Demo](https://youtu.be/sHRI0nPan20?feature=shared)

## Description

Blender MCP VXAI a powerful integration that allows you to control Blender using natural language through MCP Clients. This tool enables you to create, modify, and manipulate 3D models, animations, and scenes in Blender by simply describing what you want to do. It bridges the gap between AI language models and 3D creation, making Blender more accessible and efficient for both beginners and experienced users. This is a simple tool where the agent can let the AI agent create scripts for you while getting feedback and building the exact scene you want.

**Important:**  
If you are using `uvx blender-mcp-vxai`, please ensure you are using the latest addon file from the repository. Otherwise, make sure to use the correct version. Most setups should use version **1.0.5**, though the latest available is **1.0.6**.
- To get the exact file goto https://github.com/VxASI/blender-mcp-vxai/tree/v1.0.6 (replace the version with the version you want), and download that addon file.

## Join Discord Community
[Discord Link]([https://discord.gg/Bj7JhByPRZ](https://discord.com/invite/eswMMBghWs)
- For questions
- Suggestions
- Feedback
- Fast responses 


## Overview

Blender MCP VXAI bridges the gap between AI and 3D modeling. Whether you're a seasoned artist or just starting out, you can now:

- **Create and Modify in Real-Time:** Use plain language to instruct Blender on what to build or alter.
- **Streamline Your Workflow:** Automate complex operations and get immediate visual feedback.
- **Export Instantly:** Directly integrate your 3D models into your app or project.

**Latest Release:** v1.0.6 
*Note: Repeat the addon step with the new addon file and update your MCP server from `uv` if needed. Check the release notes for more details [Demo](https://youtu.be/3e3h6rN194I?si=E7cuDKhsHK0mcRsO).*

---

## Getting Started: Building Your 3D World

### Step 1: Prepare Your Image
- **Foundation:** Start with a basic image to serve as your project’s foundation.

### Step 2: Upload Your Image
- **Choose Your MCP Client:** Load your image into any MCP client, such as **Cursor**, **Cline**, or **Windsurf**.

### Step 3: Define Your Creative Prompt
- **Example Prompt:**
  ```plaintext
  Create this in 3D. I've given you my insane architectural plans—make it as pretty as you can! :)
  ```
- **Vision:** Describe your concept in natural language and watch your vision transform into a 3D scene.

### Step 4: Refine With Natural Language
- **Iterate:** Continue refining your scene using plain English until it perfectly matches your vision.

### Step 5: Export and Integrate
- **Command Prompt:**
  ```plaintext
  Export this scene in this project in .gib format, then create a ThreeJS app and use this as my world. Set it up as a server to avoid file-loading issues. I want to roam around this world freely—go wild!
  ```
- **Integration:** Seamlessly incorporate your 3D creation into your interactive application.

### Step 6: Experience and Enhance
- **Explore:** Enjoy and navigate through your fully-realized 3D world.
- **Polish:** Apply final tweaks to enhance the beauty and functionality of your creation.

---

## Features

- **Natural Language Commands:** Effortlessly control Blender with everyday language.
- **Seamless MCP Integration:** Work within your preferred MCP clients.
- **AI-Driven Automation:** Simplify complex 3D operations with smart text instructions.
- **Enhanced Workflow:** Transform your creative process with forward-thinking AI assistance.

---

## Installation

### Prerequisites

- **Blender**
- **Python 3.8+**

### Step 1: Install UV

UV is essential to run the MCP server.

- **macOS:**
  ```bash
  brew install uv
  ```
- **Windows/Linux:**
  ```bash
  pip install uv
  ```

### Step 2: Configure Your Environment

For the latest version, use the command: `uvx blender-mcp-vxai` (you dont need to put == version number)

#### For Cursor:
1. Click **"+ Add new Server"**.
2. Configure with:
   - **Name:** `blender-mcp`
   - **Command:** `uvx blender-mcp-vxai==1.0.5`

#### For Claude Desktop:
1. Go to **Claude > Settings > Developer > Edit Config**.
2. Open `claude_desktop_config.json` and add:
   ```json
   {
       "mcpServers": {
           "blender": {
               "command": "uvx",
               "args": [
                   "blender-mcp-vxai==1.0.5"
               ]
           }
       }
   }
   ```

### Step 3: Install the Blender Addon

1. **Download:** Get the `blender_mcp_addon.py` file.
2. **Open Blender:** Navigate to **Edit > Preferences > Add-ons**.
3. **Install Addon:** Click “Install from Disk” and select the `blender_mcp_addon.py` file.
4. **Enable:** Check the box next to “Blender MCP” to enable it.
5. **Launch MCP Server:** In Blender’s 3D View sidebar (press N if hidden), open the "BlenderMCP" tab and start the MCP server.

---

## Usage & Tools

**Available Tools:**

- **Object Creation:** Generate primitives, import models, or design complex shapes.
- **Modeling:** Modify meshes, apply modifiers, and sculpt with precision.
- **Materials:** Create and assign textures, shaders, and materials.
- **Animation:** Set keyframes, animate properties, and configure rigging.
- **Rendering:** Optimize lighting, camera setups, and render settings.
- **Scene Management:** Organize objects, collections, and entire scenes efficiently.

**Example Use Cases:**

- Transform an image into a low-poly 3D version.
- Dynamically update scene elements using detailed natural language descriptions.
- Build scenes step-by-step, modifying camera angles, colors, lighting, etc.

---

## Troubleshooting

- **Server Issues:** Verify the MCP server is running if you experience connection problems.
- **Addon Check:** Confirm that the Blender addon is properly installed and enabled.
- **Error Diagnostics:** For detailed error messages, check the Blender console.

*For older versions 1.0.3 and below, use the command:*
```bash
uvx --from blender-mcp-vxai start
```

---

## Contributing

We welcome contributions to make Blender MCP VXAI even more innovative. Please submit a Pull Request to join our forward-thinking community.

## License

Refer to the LICENSE file for full details.

---

Embrace the future of 3D modeling—explore, create, and innovate with Blender MCP VXAI.
