from .demucs import Demucs
from .hdemucs import HDemucs
from .htdemucs import HTDemucs
from omegaconf import OmegaConf

def get_model(args):
    extra = {
        "sources": list(args.training.instruments),
        "audio_channels": args.training.channels,
        "samplerate": args.training.samplerate,
        "segment": args.training.segment,
    }
    klass = {
        "demucs": Demucs,
        "hdemucs": HDemucs,
        "htdemucs": HTDemucs,
    }[args.model]
    kw = OmegaConf.to_container(getattr(args, args.model), resolve=True)
    model = klass(**extra, **kw)
    return model
