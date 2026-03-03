# models/__init__.py
# 导入所有模型类

from .lamrg import (
    _LAMRG,
    LAMRGModel,
    LAMRGModel_v7,
    LAMRGModel_v8,
    LAMRGModel_v9,
    LAMRGModel_v10,
    LAMRGModel_v11,
    BasicModel,
    # 新增CLIP增强模型
    LAMRGModel_vCLIP,
    OptimizedCLIPAlignmentModule
)

# 确保所有模型都可以被导入
__all__ = [
    '_LAMRG',
    'LAMRGModel',
    'LAMRGModel_v7',
    'LAMRGModel_v8',
    'LAMRGModel_v9',
    'LAMRGModel_v10',
    'LAMRGModel_v11',
    'BasicModel',
    'LAMRGModel_vCLIP',
    'CLIPAlignmentModule'
]

# 检查CLIP是否可用
try:
    import clip
    CLIP_AVAILABLE = True
    # print("CLIP is available for enhanced models")
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    print("LAMRGModel_vCLIP will fall back to non-CLIP mode")

# 为了向后兼容，当版本为"CLIP"时自动选择CLIP模型
def get_model_by_version(version):
    """根据版本号获取对应的模型类"""
    if version == "CLIP":
        return LAMRGModel_vCLIP
    elif version == "7":
        return LAMRGModel_v7
    elif version == "8":
        return LAMRGModel_v8
    elif version == "9":
        return LAMRGModel_v9
    elif version == "10":
        return LAMRGModel_v10
    elif version == "11":
        return LAMRGModel_v11
    elif version == "basic":
        return BasicModel
    else:
        return LAMRGModel