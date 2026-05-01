from __future__ import annotations

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from realtime_demo.demo.services import context_status, run_generation


def _int_value(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@require_GET
def index(request):
    return render(request, "demo/index.html", {"status": context_status()})


@require_GET
def status(request):
    return JsonResponse(context_status())


@require_POST
def generate(request):
    benchmark_steps_raw = request.POST.get("benchmark_steps", "25,50,100")
    benchmark_steps = []
    for item in benchmark_steps_raw.split(","):
        try:
            value = int(item.strip())
        except ValueError:
            continue
        if 1 <= value <= 500:
            benchmark_steps.append(value)
    if not benchmark_steps:
        benchmark_steps = [25, 50, 100]

    reference_index = request.POST.get("reference_index")
    reference_index_value = None if reference_index in ("", None) else _int_value(reference_index, 0)

    try:
        result = run_generation(
            target_class=request.POST.get("target_class", "guitar"),
            reference_class=request.POST.get("reference_class", "bass"),
            reference_split=request.POST.get("reference_split", "test"),
            reference_index=reference_index_value,
            uploaded_file=request.FILES.get("reference_audio"),
            seed=_int_value(request.POST.get("seed"), 2450),
            noise_scale=_float_value(request.POST.get("noise_scale"), 0.35),
            steps=_int_value(request.POST.get("steps"), benchmark_steps[-1]),
            benchmark_steps=benchmark_steps,
            config_path=request.POST.get("config_path", ""),
            ckpt_path=request.POST.get("ckpt_path", ""),
            metadata_path=request.POST.get("metadata_path", ""),
            conditioning_mode=request.POST.get("conditioning_mode", ""),
            class_names=request.POST.get("class_names", ""),
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    return JsonResponse({"ok": True, **result})
