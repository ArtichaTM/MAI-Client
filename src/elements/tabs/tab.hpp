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

    bool isVectorInBounds(const sf::Vector2f&) override;
};
