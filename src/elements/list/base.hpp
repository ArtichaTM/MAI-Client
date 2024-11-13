#pragma once
#include <vector>

#include "elements/base.hpp"

class BaseList : public SFBase {
protected:
    std::vector<SFBase*> elements;
    const float left = 0;
    const float top = 0;
    float current_left = 0;
    float current_top = 0;

    virtual void updateCurrentPosition(SFBase*) = 0;

public:
    BaseList(float left, float top);
    ~BaseList();

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    void handleEvent(const sf::Event&) override;
    void fit() override;
    void move(const float left, const float top) override;

    const std::vector<SFBase*> getElements() const;
    BaseList* addElement(SFBase*);
};
