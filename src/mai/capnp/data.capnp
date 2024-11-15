@0xc0235d78ef02ab25;

struct Vector {
    x @0 :Float32;
    y @1 :Float32;
    z @2 :Float32;
}

struct Quaternion {
    w @0 :Float32;
    x @1 :Float32;
    y @2 :Float32;
    z @3 :Float32;
}

struct RLObjectState {
    position @0 :Vector;
    speed @1 :Vector;
    rotation @2 :Vector;
    angularSpeed @3 :Vector;
}

struct GameState {
    car @0 :RLObjectState;
    ball @1 :RLObjectState;
    boostAmount @2 :UInt8;
    dead @3 :Bool;
}

struct Controls {
    throttle @0 :Float32;  # [ 0; +1]
    steer @1 :Float32;  # [-1; +1]
    pitch @2 :Float32;  # [-1; +1]
    yaw @3 :Float32;  # [-1; +1]
    roll @4 :Float32;  # [-1; +1]
    boost @5 :Bool;
    jump @6 :Bool;
}
