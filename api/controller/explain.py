import io
import pandas as pd
from fastapi import HTTPException
from matplotlib import pyplot as plt
from src.v1.explain import explain_model
from src.v1.preprocessing import preprocessor


def sync_explain(model, version, name, req):
        x, xs = preprocessor(pd.DataFrame([req.data.dict(by_alias=True)]), "no", "yes", version)
        eval_data = xs if any(k in name.lower() for k in ["logistic", "kneighbor", "knc"]) else x
        explainer = "tree" if any(k in name.lower() for k in ["forest", "tree", "boost"]) else ("linear" if any(k in name.lower() for k in ["linear", "logistic"]) else "kernel")
        
        # capture the generated PIL images directly from the execution thread
        # target_class parameter configured for index 0 to safely unpack multiclass layouts
        images = explain_model(x_test=eval_data, model=model, name=name, explainer_type=explainer, ver=version, target_class=0)
        
        if not images: 
            raise HTTPException(500, "SHAP plot generation failed.")
        
        # dynamically stitch the retrieved images vertically
        fig, axes = plt.subplots(len(images), 1, figsize=(12, 7 * len(images)))
        if len(images) == 1: 
            axes = [axes]
            
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis('off')
            
        plt.tight_layout()
        
        # write stitched compilation to stream buffer
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        img_buf.seek(0)
        return img_buf