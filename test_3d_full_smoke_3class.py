from types import SimpleNamespace

import torch

from loss import PAIRSemanticChangeLoss


def make_prediction(
    event_logits_t1,
    event_logits_t2,
):
    return SimpleNamespace(
        semantic_logits_t1=torch.tensor(
            [
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0],
            ],
            requires_grad=True,
        ),
        semantic_logits_t2=torch.tensor(
            [
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0],
            ],
            requires_grad=True,
        ),

        change_logits_t1=None,
        change_logits_t2=None,

        event_logits_t1=torch.tensor(
            event_logits_t1,
            dtype=torch.float32,
            requires_grad=True,
        ),
        event_logits_t2=torch.tensor(
            event_logits_t2,
            dtype=torch.float32,
            requires_grad=True,
        ),
    )


CLASS_NAMES = {
    0: "ground",
    1: "building",
    2: "vegetation",
    3: "clutter",
}


def make_target(
    event_t1,
    event_t2,
):
    return {
        "semantic_t1": torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
        "semantic_t2": torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),

        "event_t1": torch.tensor(
            event_t1,
            dtype=torch.long,
        ),
        "event_t2": torch.tensor(
            event_t2,
            dtype=torch.long,
        ),

        "event_valid_t1": torch.tensor(
            [True, True],
            dtype=torch.bool,
        ),
        "event_valid_t2": torch.tensor(
            [True, True],
            dtype=torch.bool,
        ),
    }


def test_illegal_logits_do_not_affect_loss():
    criterion = PAIRSemanticChangeLoss()

    # ---------------------------------------------------------
    # Prediction A:
    # Illegal logits are extremely HIGH.
    # ---------------------------------------------------------
    pred_a = make_prediction(
        event_logits_t1=[
            [5.0, 100.0, 1.0],
            [1.0, 100.0, 5.0],
        ],
        event_logits_t2=[
            [5.0, 1.0, 100.0],
            [1.0, 5.0, 100.0],
        ],
    )

    target = make_target(
        event_t1=[0, 2],
        event_t2=[0, 1],
    )

    out_a = criterion(
        prediction=pred_a,
        target=target,
        class_names=CLASS_NAMES,
    )

    # ---------------------------------------------------------
    # Prediction B:
    # Same legal logits.
    # Illegal logits are extremely LOW.
    #
    # If active support is correct, event loss A == event loss B.
    # ---------------------------------------------------------
    pred_b = make_prediction(
        event_logits_t1=[
            [5.0, -100.0, 1.0],
            [1.0, -100.0, 5.0],
        ],
        event_logits_t2=[
            [5.0, 1.0, -100.0],
            [1.0, 5.0, -100.0],
        ],
    )

    out_b = criterion(
        prediction=pred_b,
        target=target,
        class_names=CLASS_NAMES,
    )

    assert torch.allclose(
        out_a.event_t1,
        out_b.event_t1,
        atol=1e-7,
    )

    assert torch.allclose(
        out_a.event_t2,
        out_b.event_t2,
        atol=1e-7,
    )

    assert torch.allclose(
        out_a.event,
        out_b.event,
        atol=1e-7,
    )

    print(
        "[PASS] illegal logits do not affect event loss"
    )
    print(
        "       event_t1 =",
        float(out_a.event_t1.detach()),
    )
    print(
        "       event_t2 =",
        float(out_a.event_t2.detach()),
    )


def test_illegal_logits_receive_zero_gradient():
    criterion = PAIRSemanticChangeLoss()

    prediction = make_prediction(
        event_logits_t1=[
            [5.0, 100.0, 1.0],
            [1.0, 100.0, 5.0],
        ],
        event_logits_t2=[
            [5.0, 1.0, 100.0],
            [1.0, 5.0, 100.0],
        ],
    )

    target = make_target(
        event_t1=[0, 2],
        event_t2=[0, 1],
    )

    output = criterion(
        prediction=prediction,
        target=target,
        class_names=CLASS_NAMES,
    )

    output.total.backward()

    grad_t1 = prediction.event_logits_t1.grad
    grad_t2 = prediction.event_logits_t2.grad

    # T1 illegal class = Added = global ID 1.
    assert torch.allclose(
        grad_t1[:, 1],
        torch.zeros_like(grad_t1[:, 1]),
        atol=0.0,
    )

    # T2 illegal class = Removed = global ID 2.
    assert torch.allclose(
        grad_t2[:, 2],
        torch.zeros_like(grad_t2[:, 2]),
        atol=0.0,
    )

    # Legal classes must still receive gradients.
    assert grad_t1[:, 0].abs().sum() > 0
    assert grad_t1[:, 2].abs().sum() > 0

    assert grad_t2[:, 0].abs().sum() > 0
    assert grad_t2[:, 1].abs().sum() > 0

    print(
        "[PASS] illegal event logits receive zero gradient"
    )
    print("       T1 grad:")
    print(grad_t1)
    print("       T2 grad:")
    print(grad_t2)


def test_invalid_t1_target_rejected():
    criterion = PAIRSemanticChangeLoss()

    prediction = make_prediction(
        event_logits_t1=[
            [5.0, 0.0, 1.0],
            [1.0, 0.0, 5.0],
        ],
        event_logits_t2=[
            [5.0, 1.0, 0.0],
            [1.0, 5.0, 0.0],
        ],
    )

    target = make_target(
        event_t1=[
            1,  # illegal Added at T1
            2,
        ],
        event_t2=[
            0,
            1,
        ],
    )

    try:
        criterion(
            prediction=prediction,
            target=target,
            class_names=CLASS_NAMES,
        )

    except ValueError as exc:
        message = str(exc)

        assert "event_t1" in message
        assert "active support" in message

        print(
            "[PASS] invalid T1 target rejected"
        )
        print("      ", message)
        return

    raise AssertionError(
        "T1 Added target was not rejected"
    )


def test_invalid_t2_target_rejected():
    criterion = PAIRSemanticChangeLoss()

    prediction = make_prediction(
        event_logits_t1=[
            [5.0, 0.0, 1.0],
            [1.0, 0.0, 5.0],
        ],
        event_logits_t2=[
            [5.0, 1.0, 0.0],
            [1.0, 5.0, 0.0],
        ],
    )

    target = make_target(
        event_t1=[
            0,
            2,
        ],
        event_t2=[
            2,  # illegal Removed at T2
            1,
        ],
    )

    try:
        criterion(
            prediction=prediction,
            target=target,
            class_names=CLASS_NAMES,
        )

    except ValueError as exc:
        message = str(exc)

        assert "event_t2" in message
        assert "active support" in message

        print(
            "[PASS] invalid T2 target rejected"
        )
        print("      ", message)
        return

    raise AssertionError(
        "T2 Removed target was not rejected"
    )


def main():
    print("=" * 100)
    print("PAIR 3D EVENT LOSS ACTIVE-SUPPORT TEST")
    print("=" * 100)

    test_illegal_logits_do_not_affect_loss()

    print("-" * 100)

    test_illegal_logits_receive_zero_gradient()

    print("-" * 100)

    test_invalid_t1_target_rejected()

    print("-" * 100)

    test_invalid_t2_target_rejected()

    print()
    print("=" * 100)
    print("ALL EVENT LOSS ACTIVE-SUPPORT TESTS PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()