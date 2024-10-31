#pragma once
#include <vector>

#include <SFML/Graphics.hpp>

#include "../base.hpp"
#include "../interactive/button.hpp"

class Tab : public SFBase {
    std::vector<SFBase*> elements;
    bool active = false;
public:
    Button tabText;
    Tab(const std::string& title, float offset, float height, sf::Color color);
    ~Tab();
    void draw(sf::RenderWindow&) override;
    void draw(sf::RenderWindow&, bool active = false);
    void handleEvent(const sf::Event&) override;
    bool IsSwitcherInBounds(const sf::Vector2i&) const;
    bool IsSwitcherInBounds(const sf::Vector2f&) const;
    void AddElement(SFBase*);
    const std::string getName();

    void setActive(bool);
};
