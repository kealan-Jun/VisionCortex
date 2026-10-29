"""Classical dial/needle candidates. Ambiguous ends and counterweights are rejected."""
from __future__ import annotations
import math
import re

from .panel_vision import libraries

REVISION='vc-lab-dial-needle/1'


def detect_dials(image, *, limit=8):
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError('At most eight dial candidates')
    cv,np=libraries()
    height,width=image.shape[:2]
    scale=min(1,1200/max(height,width))
    gray=cv.cvtColor(cv.resize(image,None,fx=scale,fy=scale) if scale<1 else image,cv.COLOR_BGR2GRAY)
    edges=cv.Canny(cv.GaussianBlur(gray,(5,5),1),50,130)
    contours,_=cv.findContours(edges,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    candidates=[]
    for contour in contours:
        if len(contour)<60:
            continue
        (x,y),(a,b),angle=cv.fitEllipse(contour)
        if min(a,b)<.12*min(gray.shape) or max(a,b)>.95*max(gray.shape) or min(a,b)/max(a,b)<.55:
            continue
        if abs(cv.contourArea(contour))/(math.pi*a*b/4)<.8:
            continue
        # Closed rim support is needed; a short arc cannot establish a dial plane.
        radians=math.radians(angle);points=contour.reshape(-1,2)-np.array([x,y])
        rotated=points@np.array([[math.cos(radians),-math.sin(radians)],[math.sin(radians),math.cos(radians)]])
        normalized=rotated/np.array([a/2,b/2]);r=np.linalg.norm(normalized,axis=1)
        fit_error=float(np.median(abs(r-1)))
        if fit_error>.05:
            continue
        if any(math.dist([x/scale,y/scale],c['center'])<.12*max(a,b)/scale for c in candidates):
            continue
        candidates.append(dict(center=[x/scale,y/scale],axes=[a/scale,b/scale],angle=angle,
                               fit_error=fit_error,area=math.pi*a*b/(4*scale*scale)))
    candidates.sort(key=lambda c:c['area'],reverse=True)
    return dict(schema_version='vc-lab-dial-candidates/1',algorithm_revision=REVISION,status='candidate' if candidates else 'unreadable',dials=candidates[:limit])


def detect_needle(image, center, radius, *, source_sha256, max_uncertainty_degrees=3, coordinate_frame='source_image'):
    cv,np=libraries()
    if not re.fullmatch('[0-9a-f]{64}',source_sha256) or not math.isfinite(radius) or not 24<=radius<=2048 or len(center)!=2 or not all(math.isfinite(float(v)) for v in center) or not 0 < max_uncertainty_degrees <= 10:
        raise ValueError('Bounded dial geometry and source hash required')
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    cx,cy=map(float,center)
    if cx-radius<0 or cy-radius<0 or cx+radius>=gray.shape[1] or cy+radius>=gray.shape[0]:
        return dict(status='unreadable',reason='incomplete_dial',algorithm_revision=REVISION)
    # Refine the dial ellipse center against the small physical pivot/hub.
    margin=round(radius*.2);left,top=round(cx)-margin,round(cy)-margin
    hub_crop=gray[top:top+2*margin,left:left+2*margin]
    hubs=cv.HoughCircles(cv.GaussianBlur(hub_crop,(5,5),1),cv.HOUGH_GRADIENT,1,
                         max(3,radius*.04),param1=70,param2=10,minRadius=max(3,round(radius*.018)),maxRadius=max(5,round(radius*.12)))
    hub_detected=False
    if hubs is not None:
        choices=sorted(hubs[0],key=lambda c:math.dist([c[0]+left,c[1]+top],[cx,cy]))
        h=choices[0]
        if math.dist([h[0]+left,h[1]+top],[cx,cy]) <= radius*.1:
            cx,cy=float(h[0]+left),float(h[1]+top);hub_detected=True
    if cx-radius<0 or cy-radius<0 or cx+radius>=gray.shape[1] or cy+radius>=gray.shape[0]:
        return dict(status='unreadable',reason='refined_dial_outside_image',algorithm_revision=REVISION)
    # Scan outside the hub but inside the tick/number ring. Short thick weights
    # score below a continuous thin needle reaching the outer radial band.
    angles=np.arange(0,360,.25,dtype=np.float32)
    radii=np.linspace(.18,.89,180,dtype=np.float32)*radius
    theta=np.deg2rad(angles)
    mapx=(cx+np.cos(theta)[:,None]*radii[None,:]).astype(np.float32)
    mapy=(cy+np.sin(theta)[:,None]*radii[None,:]).astype(np.float32)
    polar=cv.remap(gray,mapx,mapy,cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT,borderValue=255)
    threshold,_=cv.threshold(gray[int(cy-radius):int(cy+radius)+1,int(cx-radius):int(cx+radius)+1],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    dark=polar<min(140,max(35,threshold))
    weights=np.linspace(.5,2,180)
    scores=(dark*weights).sum(axis=1)/weights.sum()
    # Continuity in eight radial bands excludes isolated letters and scale ticks.
    bands=np.array_split(dark,8,axis=1)
    continuity=np.stack([b.mean(axis=1) for b in bands],axis=1)
    valid=(continuity[:,:3].min(axis=1)>.45)&(continuity[:,3:].min(axis=1)>.35)
    scores=np.where(valid,scores,0)
    ranked=np.argsort(scores)[::-1]
    hypotheses=[]
    for index in ranked:
        if scores[index]<.5:
            break
        bearing=float(angles[index])
        if any(abs((bearing-h['angle_degrees']+180)%360-180)<10 for h in hypotheses):
            continue
        support=np.where(dark[index])[0]
        reach=float(radii[support[-1]]/radius)
        hypotheses.append(dict(angle_degrees=bearing,score=float(scores[index]),reach_fraction=reach))
        if len(hypotheses)==4:
            break
    base=dict(schema_version='vc-lab-needle-detection/1',algorithm_revision=REVISION,source_sha256=source_sha256,
              center=[str(cx),str(cy)],radius=str(radius),hypotheses=hypotheses,
              coordinate_frame=coordinate_frame,axis_hub_detected=hub_detected,
              tip_status='algorithm_candidate',accuracy='NOT_PROVEN')
    if not hypotheses:
        return dict(base,status='unreadable',reason='no_continuous_long_needle')
    best=hypotheses[0]
    if len(hypotheses)>1 and best['score']-hypotheses[1]['score']<.15:
        return dict(base,status='ambiguous',reason='multiple_long_pointer_directions')
    index=int(round(best['angle_degrees']/.25))%len(angles)
    peak=[]
    for delta in range(-40,41):
        j=(index+delta)%len(angles)
        if scores[j]>=best['score']*.8:
            peak.append(abs(delta)*.25)
    uncertainty=max(.25,max(peak,default=0))
    if uncertainty>max_uncertainty_degrees:
        return dict(base,status='ambiguous',reason='needle_angle_uncertainty')
    rad=math.radians(best['angle_degrees']);extent=radius*best['reach_fraction']
    return dict(base,status='candidate',tip=[str(cx+math.cos(rad)*extent),str(cy+math.sin(rad)*extent)],
                angle_degrees=str(best['angle_degrees']),angle_uncertainty_degrees=str(uncertainty),
                geometry_revision=REVISION,score=best['score'],reading_stage='unspecified')
