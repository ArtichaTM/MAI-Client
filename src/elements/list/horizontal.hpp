#pragma once

#include "./base.hpp"

class HorizontalList : public BaseList {
protected:
    void updateCurrentPosition(SFBase*) override;
public:
    HorizontalList(float left, float top);
    virtual sf::FloatRect getGlobalBounds() const override;
};
