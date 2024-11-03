#pragma once
#include <vector>
#include <string>

#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "elements/base.hpp"
#include "./tab.hpp"


class TabSystem : public SFBase {
    std::vector<Tab*> tabs;
    Tab* active_tab;
    sf::Font font;
    const float height;
    const sf::Color color;
    static constexpr float offset = 50.f;
public:
    TabSystem(float height, sf::Color color);
    ~TabSystem();
    operator std::string() const;

    void handleEvent(const sf::Event&) override;
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    sf::FloatRect getGlobalBounds() const override;
    virtual TabSystem* setLeft(float);
    virtual TabSystem* setTop(float);
    virtual TabSystem* setWidth(float);
    virtual TabSystem* setHeight(float);
    virtual float getLeft() const;
    virtual float getTop() const;
    virtual float getWidth() const;
    virtual float getHeight() const;

    Tab* addTab(const std::string& title);
    void firstTabInit(Tab* tab);

    void setActiveTab(Tab* tab);

    Tab* operator[](const std::string& name);
    Tab* operator[](ushort index);
};
