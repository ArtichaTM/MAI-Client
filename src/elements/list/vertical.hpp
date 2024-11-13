#pragma once

#include "./base.hpp"

class VerticalList : public BaseList {
protected:
    void updateCurrentPosition(SFBase*) override;
public:
    VerticalList(float left, float top);
    virtual sf::FloatRect getGlobalBounds() const override;
};
