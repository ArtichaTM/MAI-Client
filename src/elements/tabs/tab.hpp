#pragma once
#include <vector>

#include <SFML/Graphics.hpp>

#include "elements/base.hpp"
#include "elements/interactive/button.hpp"

class Tab : public SFBase {
    std::vector<SFBase*> elements;
    bool active = false;
public:
    Button tabText;

    Tab(const std::string&, float offset, float height, sf::Color);
    ~Tab();
    operator std::string() const;

    void handleEvent(const sf::Event&) override;
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    void AddElement(SFBase*);
    const std::string getName();
    void setActive(bool);

    sf::FloatRect getGlobalBounds() const override;
    virtual Tab* setLeft(float);
    virtual Tab* setTop(float);
    virtual Tab* setWidth(float);
    virtual Tab* setHeight(float);
    virtual float getLeft() const;
    virtual float getTop() const;
    virtual float getWidth() const;
    virtual float getHeight() const;
};
